import sys
from pathlib import Path
import os
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ai.help_folder.help import calculate_base_features, optimize_threshold

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
    vol_thresholds = df_test['dynamic_threshold'].values if 'dynamic_threshold' in df_test.columns else np.full(len(prices), 0.02)
    n = len(prices)

    equity = [1.0]
    position = 0 
    entry_price = 0
    entry_index = 0
    last_trade_index = -100
    trades = []

    for i in range(1, n):
        curr_equity = equity[-1]
        
        if position != 0:
            step_ret = (prices[i] - prices[i-1]) / prices[i-1] * position
            curr_equity *= (1 + step_ret * max_position_size)
            raw_pnl = (prices[i] - entry_price) / entry_price * position
            hold_time = i - entry_index
            curr_tp_sl = vol_thresholds[entry_index]
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
        elif i - last_trade_index > 3:
            if custom_signals is not None and custom_signals[i] != 0:
                position = custom_signals[i]
                entry_price = prices[i]
                entry_index = i
                trades.append({'type': 'ENTRY', 'index': i, 'price': prices[i]})

        equity.append(curr_equity)

    equity = np.array(equity)
    exit_trades = [t for t in trades if t['type'] == 'EXIT']
    wins = [t for t in exit_trades if t['pnl'] > 0]
    win_rate = (len(wins) / len(exit_trades) * 100) if len(exit_trades) > 0 else 0
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = np.min(drawdowns) * 100 # в процентах
    avg_pnl = np.mean([t['pnl'] for t in exit_trades]) * 100 if len(exit_trades) > 0 else 0
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

def get_percentile_thresholds(probs_move, probs_up, top_k=15):
    t_move = np.percentile(probs_move, 100 - top_k)
    mask = probs_move >= t_move
    if not any(mask): 
        t_buy = np.percentile(probs_up, 70)
        t_sell = np.percentile(probs_up, 30)
    else:
        relevant_ups = probs_up[mask]
        t_buy = np.percentile(relevant_ups, 70) 
        t_sell = np.percentile(relevant_ups, 30) 
    return t_move, t_buy, t_sell

@router.get("/run/{symbol}")
async def run_detailed_backtest(symbol: str, days: int = 100):
    # 0. Инициализируем переменные заранее, чтобы избежать ошибки "local variable"
    total_rows = 0 
    
    try:
        if len(MODELS) == 0: 
            load_models()
        
        # 1. Динамический расчет периода загрузки
        # Чтобы протестировать 600 дней + 200 дней прогрева, нужно минимум 4 года
        needed_years = (days + 250) // 252 + 1
        period_str = f"{needed_years}y"
        
        print(f"Загрузка данных для {symbol}, период: {period_str}")
        data = yf.download(symbol.upper(), period=period_str, progress=False) 
        
        if data.empty: 
            raise HTTPException(404, detail=f"Тикер {symbol} не найден")
            
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
       
        # 2. Расчет признаков
        df = calculate_base_features(data.dropna())
        df = df.dropna()
        
        # 3. Теперь безопасно определяем количество строк
        total_rows = len(df)
        
        # Если пользователь просит больше, чем есть в истории Yahoo
        if total_rows < (days + 150):
            actual_max_days = total_rows - 151
            if actual_max_days <= 0:
                raise HTTPException(400, detail=f"Недостаточно истории. Доступно всего {total_rows} дней.")
            # Вместо ошибки просто уменьшаем количество дней до доступного максимума
            days = actual_max_days

        test_start_idx = total_rows - days
        calibration_start_idx = test_start_idx - 150 

        all_results = {}
        chart_data_models = {}
        test_df = df.iloc[test_start_idx:].copy()

        # 4. Цикл по моделям (код остается таким же, как был)
        for m_name in ['gru', 'lstm', 'cnn']:
            if f"{m_name}_stage1" not in MODELS: continue
            
            seq_len = CONFIG.get(m_name, {}).get('sequence_length', 20)
            
            calib_probs_move, calib_probs_up = [], []
            for i in range(calibration_start_idx, test_start_idx):
                history = df.iloc[i - seq_len + 1 : i + 1]
                res = predict(history, m_name)
                calib_probs_move.append(res['prob_move'])
                calib_probs_up.append(res['prob_up'])
            
            t_move_limit, t_buy_limit, t_sell_limit = get_percentile_thresholds(
                np.array(calib_probs_move), np.array(calib_probs_up), top_k=15
            )

            final_signals = []
            probs_to_chart = [] 
            in_position = 0 
            
            for i in range(days):
                curr_idx = test_start_idx + i
                history = df.iloc[curr_idx - seq_len + 1 : curr_idx + 1]
                res = predict(history, m_name)
                p_move, p_up = res['prob_move'], res['prob_up']
                probs_to_chart.append(p_up)
                
                signal = 0
                if in_position == 0:
                    if p_move >= t_move_limit:
                        if p_up >= t_buy_limit: signal = 1; in_position = 1
                        elif p_up <= t_sell_limit: signal = -1; in_position = -1
                elif in_position == 1 and p_up < 0.48: signal = -1; in_position = 0
                elif in_position == -1 and p_up > 0.52: signal = 1; in_position = 0
                final_signals.append(signal)

            stats = production_backtest(test_df, None, None, custom_signals=final_signals, horizon=5)

            all_results[m_name] = {
                "final_balance": round(10000 * stats['equity'][-1], 2),
                "profit_pct": round(stats['total_return'], 2),
                "win_rate": stats.get('win_rate', 0),
                "max_dd": stats.get('max_dd', 0),
                "avg_pnl": stats.get('avg_pnl', 0),
                "sharpe": stats.get('sharpe', 0),
                "num_trades": stats['num_trades'],
                "trades": [] # Для краткости
            }
            chart_data_models[m_name] = {
                "buy": [test_df['Close'].iloc[idx] if final_signals[idx] == 1 else None for idx in range(days)],
                "sell": [test_df['Close'].iloc[idx] if final_signals[idx] == -1 else None for idx in range(days)],
                "predicted": [round(test_df['Close'].iloc[idx]*(1+(p-0.5)*0.04), 2) for idx, p in enumerate(probs_to_chart)]
            }

        return {
            "results": all_results,
            "chart_data": {
                "dates": [str(d.date()) for d in test_df.index],
                "real_prices": test_df['Close'].round(2).tolist(),
                "models": chart_data_models
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        # total_rows тут уже точно определена как 0 или число
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)} (Rows: {total_rows})")