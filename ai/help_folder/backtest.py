import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def backtest(results, history_stages=None, strategy_name="Strategy",
             thresholds=None, save_dir="../models"):
    
    if len(results.get('equity', [])) > 0 and np.all(np.isfinite(results['equity'])):
        plt.figure(figsize=(14, 6))
        plt.plot(results['equity'], label=strategy_name, linewidth=2, color='green')
        plt.plot([1, len(results['equity'])], [1, 1], 'k--', label='Buy & Hold', alpha=0.3)
        plt.xlabel('Trading Days')
        plt.ylabel('Equity')
        title = f'{strategy_name} Backtest'
        if thresholds:
            title += " " + " ".join([f"{k}={v:.2f}" for k, v in thresholds.items()])
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        equity_path = f"{save_dir}/equity_curve_{strategy_name.lower().replace(' ', '_')}.png"
        plt.savefig(equity_path, dpi=150)
    
    if history_stages:
        plt.figure(figsize=(12 * len(history_stages), 4))
        for i, history in enumerate(history_stages, 1):
            if history is None:
                plt.text(0.5, 0.5, f'Stage {i}: No Training\n(No data passed filter)', 
                        ha='center', va='center', fontsize=12, color='red')
                plt.title(f'Stage {i} Loss (Skipped)')
                continue
            
            plt.subplot(1, len(history_stages), i)
            plt.plot(history.history['loss'], label=f'Stage{i} Train')
            plt.plot(history.history['val_loss'], label=f'Stage{i} Val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Stage {i} Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        training_path = f"{save_dir}/training_history_{strategy_name.lower().replace(' ', '_')}.png"
        plt.savefig(training_path, dpi=150)

def production_backtest(
    df_test, 
    probs1, 
    probs2,
    threshold1=0.8, 
    threshold2=0.7,
    commission=0.001, 
    slippage=0.0005,
    max_position_size=0.10, 
    horizon=5
):
    prices = df_test['Close'].values
    vol_thresholds = df_test['dynamic_threshold'].values 
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
            raw_pnl = (prices[i] - entry_price) / entry_price * position
            hold_time = i - entry_index
            curr_tp_sl = vol_thresholds[entry_index]
            
            if raw_pnl >= curr_tp_sl or raw_pnl <= -curr_tp_sl or hold_time >= horizon:
                final_pnl = (raw_pnl - (commission + slippage)) * max_position_size
    
                trades.append({
                    'pnl': final_pnl, 
                    'type': 'EXIT', 
                    'hold': hold_time,
                    'direction': position
                })
                position = 0
                last_trade_index = i
            else:
                daily_ret = (prices[i] - prices[i-1]) / prices[i-1] * position
                curr_equity *= (1 + daily_ret * max_position_size)

        elif i - last_trade_index > 3:
            p_move = probs1[i]
            p_up = probs2[i]
            
            if p_move > threshold1 and abs(p_up - 0.5) > (threshold2 - 0.5):
                position = 1 if p_up > 0.5 else -1
                entry_price = prices[i]
                entry_index = i

        equity.append(curr_equity)

    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]

    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_dd = np.max(drawdowns)
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
    
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    avg_pnl = np.mean([t['pnl'] for t in trades]) if trades else 0

    return {
        'equity': equity,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'total_return': equity[-1] - 1,
        'trades': trades,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl, 
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }


def get_adaptive_threshold(regime_score=None, vol_regime=None, base_threshold=0.75, low=0.50, high=0.90):
    threshold = base_threshold
    if regime_score is not None:
        if regime_score > 0.7:  
            threshold -= 0.05
        elif regime_score < 0.3:
            threshold += 0.05
    if vol_regime is not None:
        if vol_regime == 1:      
            threshold += 0.05
        else:                   
            threshold -= 0.02
    return np.clip(threshold, low, high)

def filter_signals(preds, probs, regime_features=None, PROB_THRESHOLD_BASE=0.50):
    n = len(preds)
    filtered_preds = preds.copy()
    filter_mask = np.ones(n, dtype=bool)
    
    for i in range(n):
        signal = preds[i]
        confidence = np.max(probs[i])
        if signal == 0:
            continue
    
        regime_score = regime_features.get('regime_score', [None] * n)[i] if 'regime_score' in regime_features else None
        vol_regime = regime_features.get('vol_regime', [None] * n)[i] if 'vol_regime' in regime_features else None
        threshold = get_adaptive_threshold(regime_score, vol_regime, base_threshold=PROB_THRESHOLD_BASE)
 
        if confidence < threshold:
            filtered_preds[i] = 0
            filter_mask[i] = False
    
    final_mask = (filtered_preds == 1)
    return filtered_preds, final_mask