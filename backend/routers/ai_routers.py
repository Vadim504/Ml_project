import sys
from pathlib import Path
import os
from .stock import get_stock_data_internal  # Внутренняя функция для получения данных без HTTP
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ai.help_folder.help import calculate_base_features 
project_root = Path(__file__).parent.parent.parent  
sys.path.insert(0, str(project_root))

from ai.predictor import (
    MODELS, SCALERS, CONFIG,   
    prepare_input,     
    predict, predict_sequence          
)

from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd
import numpy as np

router = APIRouter()

@router.get("/predict/{symbol}")
async def get_prediction(symbol: str, model_type: str = "model_gru"): # Используем model_type как в JS
    try:
        data = yf.download(symbol.upper(), period="2y", progress=False)
        if data.empty:
            raise HTTPException(status_code=404, detail="Тикер не найден или нет данных")
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        df = data.copy()
        df = df.dropna()
        df = calculate_base_features(df)  
        result = predict(df, model_type=model_type) 
        
        return {
            "symbol": symbol.upper(),
            "model": model_type,
            "signal": result['signal'],
            "prob_move": round(result['prob_move'], 3) if 'prob_move' in result else 0,
            "prob_up": round(result['prob_up'], 3) if 'prob_up' in result else 0,
            "confidence": round(result['confidence'], 3) if 'confidence' in result else 0,
            "data_points": len(df),
            "last_price": round(float(df['Close'].iloc[-1]), 2),
            "last_date": str(df.index[-1].date())
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка в эндпоинте predict: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")
    
@router.get("/predict-future/{symbol}")
async def get_future_forecast(symbol: str, model_type: str = "gru", days: int = 7):
    try:
       
        model_type = model_type.replace('model_', '')

        # 2. Получаем данные (берем 1 год, чтобы ИИ хватило данных)
        from .stock import get_stock_data_internal
        df = await get_stock_data_internal(symbol)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Биржа не отдала данные")

        # Считаем прогноз
        forecast_prices = predict_sequence(df, model_type, days)

        # 4. Считаем «подгонку» под историю (фиолетовая линия на прошлом)
        history_preds = []
        # Берем последние 30-50 дней истории для наложения
        actual_data_tail = df.tail(50) 
        
        # Окно для модели
        seq_len = CONFIG.get(model_type, {}).get('sequence_length', 80)
        
        for i in range(len(actual_data_tail)):
            # Берем окно ПЕРЕД этой точкой
            idx = len(df) - len(actual_data_tail) + i
            window = df.iloc[idx - seq_len : idx]
            
            if len(window) == seq_len:
                X = prepare_input(calculate_base_features(window), model_type)
                model = MODELS[f'{model_type}_stage2']
                p = model(X, training=False).numpy()[0, 0]
                
                # Конвертируем вероятность в цену
                last_p = float(window['Close'].iloc[-1])
                change = (float(p) - 0.5) * 0.04
                history_preds.append(float(last_p * (1 + change)))
            else:
                history_preds.append(None)

        last_date = df.index[-1]
        dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]

        return {
            "status": "success",
            "prices": [float(p) for p in forecast_prices],
            "history_prices": [float(p) if p is not None else None for p in history_preds],
            "dates": dates,
            "signal": "BUY" if forecast_prices[-1] > forecast_prices[0] else "SELL"
        }

    except Exception as e:
        import traceback
        print(f"❌ ОШИБКА В РОУТЕРЕ: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/predict-history/{symbol}")
async def get_historical_predictions(symbol: str, model_type: str = "gru"):
    try:
        model_type = model_type.replace('model_', '')
        df = await get_stock_data_internal(symbol)
        if df is None or len(df) < 70:
            raise HTTPException(status_code=400, detail="Недостаточно данных для теста")

        test_df = df.tail(50) 
        actual_prices = test_df['Close'].tolist()
        dates = [d.strftime("%Y-%m-%d") for d in test_df.index]
        history_preds = []
        
        for i in range(len(test_df)):
            end_idx = len(df) - len(test_df) + i
            window_df = df.iloc[end_idx-60 : end_idx]
            
            if len(window_df) < 60:
                history_preds.append(None)
                continue
                
            X = prepare_input(calculate_base_features(window_df), model_type)
            pred = MODELS[f'{model_type}_stage2'].predict(X, verbose=0)[0, 0]
            last_p = float(window_df['Close'].iloc[-1])
            change = (float(pred) - 0.5) * 0.04
            history_preds.append(round(last_p * (1 + change), 2))

        return {
            "dates": dates,
            "actual": actual_prices,
            "predicted": history_preds
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))