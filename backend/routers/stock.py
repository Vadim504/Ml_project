# backend/app/routers/stock.py
from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd
import numpy as np

router = APIRouter()

# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (Сердце этого файла) ---
async def _download_and_clean(symbol: str, period: str):
    try:
        data = yf.download(symbol.upper(), period=period, progress=False)
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            df = pd.DataFrame(index=data.index)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = data[col].iloc[:, 0]
        else:
            # Выбираем только нужные колонки, чтобы не захламлять память
            df = data[["Open", "High", "Low", "Close", "Volume"]].copy()

        return df.dropna()
    except Exception as e:
        print(f"❌ Ошибка загрузки {symbol}: {e}")
        return None

# --- 1. ЭНДПОИНТ ДЛЯ ФРОНТЕНДА ---
@router.get("/data/{symbol}")
async def get_stock_data(symbol: str, period: str = "1y"):
    df = await _download_and_clean(symbol, period)
    
    if df is None:
        raise HTTPException(status_code=404, detail=f"Данные для {symbol} не найдены")

    # Переименовываем только для отправки в браузер
    df = df.rename(columns={
        "Open": "1. open", "High": "2. high", "Low": "3. low",
        "Close": "4. close", "Volume": "5. volume"
    })

    # Формируем JSON
    time_series = df.to_dict(orient="index")
    result = {
        date.strftime("%Y-%m-%d"): {
            k: (int(v) if k == "5. volume" else round(float(v), 2))
            for k, v in row.items()
        }
        for date, row in time_series.items()
    }

    return {"time_series": result, "symbol": symbol.upper()}

# --- 2. ФУНКЦИЯ ДЛЯ ИИ-ПРЕДСКАЗАТЕЛЯ ---
async def get_stock_data_internal(symbol: str):
    """Используется в ai_routers.py. Всегда берет 1 год для работы нейросети."""
    return await _download_and_clean(symbol, period="1y")