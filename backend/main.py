# main.py
from pathlib import Path
import pandas as pd
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import yfinance as yf

from ai.predictor import load_models
sys.path.insert(0, str(Path(__file__).parent))
from routers import stock, ai_routers, backtest

app = FastAPI(title="Stock Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(ai_routers.router, prefix="/api", tags=["AI"])
app.include_router(stock.router, prefix="/api/stock", tags=["Stock"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/api/stock-data/{symbol}")
async def get_stock_data(symbol: str, period: str = "1y"):
    try:
        # 1. Скачиваем данные
        data = yf.download(symbol.upper(), period=period, progress=False)
        
        if data.empty:
            raise HTTPException(404, f"Нет данных для {symbol}")
        
        # --- ИСПРАВЛЕНИЕ ТУТ: Убираем многоуровневые колонки ---
        # Если yfinance вернул MultiIndex (например, с уровнем тикера), берем только названия колонок
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # -----------------------------------------------------

        # Переименовываем колонки (теперь они гарантированно обычные строки)
        data = data.rename(columns={
            "Open": "1. open", "High": "2. high", 
            "Low": "3. low", "Close": "4. close", "Volume": "5. volume"
        }).dropna()
        
        time_series = {
            date.strftime("%Y-%m-%d"): {
                str(k): (round(v, 2) if k != "5. volume" else int(v)) # Добавили str(k) для страховки
                for k, v in row.items()
            }
            for date, row in data.to_dict(orient="index").items()
        }
        
        # Берем последние 60 точек
        latest_dates = sorted(time_series.keys())[-60:]
        latest = {d: time_series[d] for d in latest_dates}
        
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        
        return {
            "symbol": symbol.upper(),
            "currency": info.get("currency", "USD"),
            "name": info.get("shortName", symbol.upper()),
            "time_series": latest,
            "data_points": len(latest)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка: {str(e)}")

project_root = Path(__file__).parent.parent
FRONTEND_DIR = project_root / "frontend" / "public_html"

if FRONTEND_DIR.exists():
    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def serve_spa(full_path: str):
        # Если путь пустой - отдаем главную
        if not full_path:
            return FileResponse(str(FRONTEND_DIR / "index.html"))

        # Если запрашивается конкретный HTML файл (например /backtest.html)
        requested_file = FRONTEND_DIR / full_path
        if requested_file.exists() and requested_file.suffix == ".html":
            return FileResponse(str(requested_file))
            
        # Для всех остальных путей (SPA роутинг) отдаем главную
        return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": len(ai_routers.MODELS) > 0
    }