# ai/predictor.py

import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from ai.features import FEATURES
from ai.help_folder.help import optimize_threshold

MODELS = {}     
SCALERS = {}     
CONFIG = {}      

def load_models():
    models_dir = Path(__file__).parent.parent / 'ai' / 'models'
    mtypes = ['model_cnn', 'model_lstm', 'model_gru']
    for mtype in mtypes:
        short_name = mtype.replace('model_', '') 
        
        mdir = models_dir / mtype / 'strategy' 
        files = {
            'stage1': mdir / 'model_stage1.h5',
            'stage2': mdir / 'model_stage2.h5',
            'scaler': mdir / 'scaler.pkl',
            'params': mdir / 'params.pkl'
        }
        MODELS[f'{short_name}_stage1'] = load_model(str(files['stage1']))
        MODELS[f'{short_name}_stage2'] = load_model(str(files['stage2']))
        SCALERS[short_name] = joblib.load(str(files['scaler']))
        CONFIG[short_name] = joblib.load(str(files['params']))

def prepare_input(df, model_type='gru'):
    mtype = model_type.replace('model_', '')
    scaler = SCALERS.get(mtype)
    cfg = CONFIG.get(mtype, {})
    seq_len = cfg.get('sequence_length', 80)
    
    if scaler is None:
        raise ValueError(f"❌ Скалер для {mtype} не найден в словаре SCALERS")
    
    target_features = [f.lower().strip().replace(' ', '_') for f in FEATURES]
    df_clean = df.copy()
    df_clean.columns = [str(c).lower().strip().replace(' ', '_') for c in df_clean.columns]

    for col in target_features:
        if col not in df_clean.columns:
            df_clean[col] = 0.0

    df_clean = df_clean.fillna(0) 
    final_df = df_clean[target_features]
    data = final_df.values

    n_expected = scaler.n_features_in_ 
    if data.shape[1] != n_expected:
        if data.shape[1] > n_expected:
            data = data[:, :n_expected]
        else:
            diff = n_expected - data.shape[1]
            data = np.hstack([data, np.zeros((data.shape[0], diff))])

    if data.shape[0] < seq_len:
        pad_size = seq_len - data.shape[0]
        padding = np.tile(data[0], (pad_size, 1))
        data = np.vstack([padding, data])
    
    data = data[-seq_len:]
    scaled_data = scaler.transform(data)
    return scaled_data.reshape(1, seq_len, -1)

def predict_sequence(df, model_type='gru', days=7):
    model_type = model_type.replace('model_', '')
    current_df = df.copy().tail(120)
    future_prices = []
    
    try:
        for _ in range(days):
            X = prepare_input(current_df, model_type)
            
            model = MODELS[f'{model_type}_stage2']
            raw_pred = model(X, training=False).numpy()[0, 0]
            pred_val = float(raw_pred)
            last_price = float(current_df['Close'].iloc[-1])
            
            if pred_val <= 1.0: 
                change = (pred_val - 0.5) * 0.04 
                next_price = last_price * (1 + change)
            else: 
                next_price = pred_val

            last_date = current_df.index[-1]
            if not isinstance(last_date, pd.Timestamp):
                last_date = pd.to_datetime(last_date)
                
            new_row = pd.DataFrame({
                'Open': [last_price],
                'High': [next_price if next_price > last_price else last_price],
                'Low': [next_price if next_price < last_price else last_price],
                'Close': [next_price],
                'Volume': [0] 
            }, index=[last_date + pd.Timedelta(days=1)])
            
            current_df = pd.concat([current_df, new_row])
            future_prices.append(round(next_price, 2))
            
        return future_prices

    except Exception as e:
        print(f"❌ Ошибка в цикле предсказания: {e}")
        return [float(df['Close'].iloc[-1])] * days

def predict(df, model_type='gru'):
    mtype = model_type.replace('model_', '')
    try:
        X = prepare_input(df, mtype)
        cfg = CONFIG.get(mtype, {})
        if model_type=='gru':
            t1 = cfg.get('threshold_stage1', 0.20)
            t_buy = cfg.get('threshold_buy', 0.54) 
            t_sell = cfg.get('threshold_sell', 0.50)
        elif model_type=='lstm':
            t1 = cfg.get('threshold_stage1', 0.15)
            t_buy = cfg.get('threshold_buy', 0.58) 
            t_sell = cfg.get('threshold_sell', 0.60)
        else:
            t1 = cfg.get('threshold_stage1', 0.45)
            t_buy = cfg.get('threshold_buy', 0.65) 
            t_sell = cfg.get('threshold_sell', 0.40)


        p_move = float(MODELS[f'{mtype}_stage1'].predict(X, verbose=0)[0, 0])
        p_up = float(MODELS[f'{mtype}_stage2'].predict(X, verbose=0)[0, 0])
    
        # Логика сигналов
        sig = 'hold'
        if p_move >= t1 and p_up >= t_buy:
            sig = 'buy'
        elif p_up < t_sell:
            sig = 'sell'
        
        if sig == 'buy':
            conf = float(p_up)
        elif sig == 'sell':
            conf = float(1 - p_up)
        else:
            conf = float(1 - p_move)

        return {
            'signal': sig, # Это будет "рекомендация", но бэктест решит сам
            'prob_move': p_move,
            'prob_up': p_up,
            'confidence': conf,
            'stage': 2 if p_move >= t1 else 1, # Добавляем для фронтенда
            'last_price': float(df['Close'].iloc[-1])
        }
        
    except Exception as e:
        print(f"❌ Predict error for {mtype}: {e}")
        return {
            'signal': 'hold',
            'prob_move': 0.0,
            'prob_up': 0.5,
            'confidence': 0.0,
            'stage': 1,
            'error': str(e),
        }

load_models()