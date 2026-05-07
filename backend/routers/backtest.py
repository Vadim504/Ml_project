import sys
from pathlib import Path
import os
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ai.help_folder.help import calculate_base_features, optimize_threshold
from ai.help_folder.backtest import get_adaptive_threshold
project_root = Path(__file__).parent.parent.parent  
sys.path.insert(0, str(project_root))

from ai.predictor import (
    MODELS, SCALERS, CONFIG,   
    prepare_input,     
    predict             
)

from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd
import numpy as np
from ai.predictor import load_models

router = APIRouter()


def production_backtest(df_test, probs1, probs2, custom_signals=None, 
                        commission=0.001, slippage=0.0005, max_position_size=1.0, horizon=5):
    prices = df_test['Close'].values
    # Используем динамический порог волатильности, если он есть, иначе 2%
    vol_thresholds = df_test['dynamic_threshold'].values if 'dynamic_threshold' in df_test.columns else np.full(len(prices), 0.02)
    n = len(prices)

    equity = [1.0]
    position = 0 # 0 - кэш, 1 - лонг, -1 - шорт (если поддерживается)
    entry_price = 0
    entry_index = 0
    last_trade_index = -100
    trades = []

    for i in range(1, n):
        curr_equity = equity[-1]
        
        if position != 0:
            # 1. Считаем изменение капитала на текущем шаге
            step_ret = (prices[i] - prices[i-1]) / prices[i-1] * position
            curr_equity *= (1 + step_ret * max_position_size)
            
            # 2. Проверяем условия выхода
            raw_pnl = (prices[i] - entry_price) / entry_price * position
            hold_time = i - entry_index
            curr_tp_sl = vol_thresholds[entry_index]
            
            # Добавили условие: выходим если TP/SL, если время вышло ИЛИ если пришел сигнал SELL (-1)
            model_exit_signal = (custom_signals is not None and custom_signals[i] == -1)
            
            if raw_pnl >= curr_tp_sl or raw_pnl <= -curr_tp_sl or hold_time >= horizon or model_exit_signal:
                trades.append({
                    'pnl': (raw_pnl - (commission + slippage)) * max_position_size, 
                    'type': 'EXIT', 
                    'index': i,
                    'price': prices[i]
                })
                position = 0
                last_trade_index = i
        
        # Логика входа (только если мы не в позиции и прошел период охлаждения)
        elif i - last_trade_index > 3:
            if custom_signals is not None and custom_signals[i] == 1:
                position = 1
                entry_price = prices[i]
                entry_index = i
                trades.append({'type': 'ENTRY', 'index': i, 'price': prices[i]})

        equity.append(curr_equity)

    # --- РАСЧЕТ МЕТРИК (Вне цикла!) ---
    equity = np.array(equity)
    exit_trades = [t for t in trades if t['type'] == 'EXIT']
    
    # Win Rate
    wins = [t for t in exit_trades if t['pnl'] > 0]
    win_rate = (len(wins) / len(exit_trades) * 100) if len(exit_trades) > 0 else 0

    # Max Drawdown
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = np.min(drawdowns) * 100 # в процентах

    # Average PnL
    avg_pnl = np.mean([t['pnl'] for t in exit_trades]) * 100 if len(exit_trades) > 0 else 0

    # Sharpe Ratio
    returns = np.diff(equity) / (equity[:-1] + 1e-9)
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0
    
    return {
        'equity': equity,

        'win_rate': round(win_rate, 2),
        'max_dd': round(max_dd, 2),
        'avg_pnl': round(avg_pnl, 2),
        'sharpe': round(sharpe, 2),
        'total_return': round((equity[-1] - 1) * 100, 2),

        'trades': trades,
        'num_trades': len(exit_trades)
    }

@router.get("/run/{symbol}")
async def run_detailed_backtest(symbol: str, days: int = 100):
    try:
        if len(MODELS) == 0: load_models()
        
        # 1. Загрузка данных (теперь всегда берем с запасом)
        warmup_days = 150 + 80 + 100
        total_days_needed = days + warmup_days
        years_needed = (total_days_needed // 252) + 1
        period_str = f"{years_needed}y" if years_needed < 10 else "max"
        data = yf.download(symbol.upper(), period=period_str, progress=False)

        if data.empty: raise HTTPException(404, detail="Данные не найдены")
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
       
        df = calculate_base_features(data.dropna())
        
        df['regime_score'] = df['Close'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-9), raw=False
        )
        std_20 = df['Close'].pct_change().rolling(20).std()
        std_100 = df['Close'].pct_change().rolling(100).std()
        df['vol_regime'] = (std_20 > std_100).astype(int)
        
        df = df.dropna().fillna(0.02)
        total_rows = len(df)

        # Проверка лимитов истории
        if total_rows < (days + 230):
            days = total_rows - 230
            if days <= 0: raise HTTPException(400, detail="Недостаточно истории")

        test_start_idx = total_rows - days
        val_start_idx = test_start_idx - 150 

        all_results = {}
        chart_data_models = {}
        test_df = df.iloc[test_start_idx:].copy()

        for m_name in ['gru', 'lstm', 'cnn']:
            if f"{m_name}_stage1" not in MODELS: continue
            seq_len = CONFIG.get(m_name, {}).get('sequence_length', 80)
            
            
            X_val, y_val = [], []
            for i in range(val_start_idx, test_start_idx):
                history_window = df.iloc[i - seq_len : i]
                X_single = prepare_input(history_window, model_type=m_name) 
                X_val.append(X_single[0])
                target = 1 if df['Close'].iloc[i] > df['Close'].iloc[i-1] else 0
                y_val.append(target)
            
            if m_name=='gru':
                t_base, _, _ = optimize_threshold(MODELS[f"{m_name}_stage2"], np.array(X_val), np.array(y_val), metric='f1', low=0.48, high=0.75)
            elif m_name=='lstm':
                t_base, _, _ = optimize_threshold(MODELS[f"{m_name}_stage2"], np.array(X_val), np.array(y_val), metric='f1',low=0.55, high=0.80)
            else:
                t_base, _, _ = optimize_threshold(MODELS[f"{m_name}_stage2"], np.array(X_val), np.array(y_val), metric='f1', low=0.75, high=0.9)
        
            final_signals = []
            prob_history = []
            probs_to_chart = [] 
            in_position = False
            last_action_idx = -10

            for i in range(days):
                curr_idx = test_start_idx + i
                history = df.iloc[curr_idx - seq_len + 1 : curr_idx + 1]
                res = predict(history, m_name)
            
                prob_history.append(res['prob_up'])
                probs_to_chart.append(res['prob_up'])
                if len(prob_history) > 3: prob_history.pop(0)
                p_smooth = sum(prob_history) / len(prob_history)
                
                t_buy = get_adaptive_threshold(df['regime_score'].iloc[curr_idx], df['vol_regime'].iloc[curr_idx], t_base)
                if m_name == 'lstm':
                    t_sell = t_base * 0.95
                elif m_name == 'gru':
                    t_sell = t_base * 0.97
                elif m_name == 'cnn':
                    t_sell = t_base * 0.85
                signal = 0
    
                if not in_position and res['prob_move'] > 0.10 and p_smooth >= t_buy:
                    if (i - last_action_idx) > 3: # Не чаще раз в 5 дней
                        signal = 1
                        in_position = True
                        last_action_idx = i
                elif in_position and p_smooth < t_sell:
                    signal = -1
                    in_position = False
                    last_action_idx = i
                
                final_signals.append(signal)

            stats = production_backtest(test_df, np.ones(days), np.ones(days), custom_signals=final_signals,horizon=1000)

            trades_list = []
            for t in stats.get('trades', []):
                trades_list.append({
                    "date": str(test_df.index[t['index']].date()),
                    "type": "BUY" if t['type'] == 'ENTRY' else "SELL",
                    "price": round(t['price'], 2)
                })

            all_results[m_name] = {
                "final_balance": round(10000 * stats['equity'][-1], 2),
                "profit_pct": round(stats['total_return'], 2),
                
                "win_rate": stats.get('win_rate', 0),
                "max_dd": stats.get('max_dd', 0),
                "avg_pnl": stats.get('avg_pnl', 0),
                "sharpe": stats.get('sharpe', 0),

                "num_trades": stats['num_trades'],
                "trades": trades_list
            }
            
            chart_data_models[m_name] = {
                "buy": [test_df['Close'].iloc[idx] if final_signals[idx] == 1 else None for idx in range(days)],
                "sell": [test_df['Close'].iloc[idx] if final_signals[idx] == -1 else None for idx in range(days)],
                "predicted": [round(test_df['Close'].iloc[idx]*(1+(p-0.5)*0.04), 2) for idx, p in enumerate(probs_to_chart)] # упрощенно
            }

        return {
            "results": all_results,
            "chart_data": {
                "dates": [str(d.date()) for d in test_df.index],
                "real_prices": test_df['Close'].round(2).tolist(),
                "models": chart_data_models
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))
