# FEATURES = [
#     'Return','Return_5d','Return_10d','Return_20d',
#     'Mom_5','Mom_10','Mom_20','Mom_Ratio','ROC_5','ROC_10','Momentum',
#     'Vol_10d','Vol_20d','Vol_Regime','Volatility_Ratio',
#     'RSI','MACD','MACD_Signal','MACD_Hist',
#     'BB_Ratio',
#     'SMA_50','SMA_200','Trend_50','Trend_200',
#     'Trend_Regime','Trend_Strength','Trend_50_Ratio','Trend_200_Ratio',
#     'MA_Ratio',
#     'ZScore_20',
#     'Volume_Ratio','Volume_Change','OBV','OBV_Change',
#     'Sentiment_Score',
#     'Return_lag1','Return_lag3','Return_lag5','Vol_lag1','Vol_lag3',
#     'Vol_lag5','Mom_lag1','Mom_lag3','Mom_lag5',
#     'Price_to_High_20','Price_to_Low_20'
# ]

FEATURES = [

# --- Returns ---
'Return',
'Return_5d',
'Return_10d',

# --- Momentum ---
'Momentum',
'ROC_10',

# --- Volatility ---
'Vol_20d',
'Volatility_Ratio',

# --- Oscillators ---
'RSI',
'MACD_Hist',
'BB_Ratio',

# --- Trend ---
'Trend_50_Ratio',
'Trend_200_Ratio',

# --- Mean Reversion ---
'ZScore_20',

# --- Volume ---

# --- Range position ---
'Price_to_High_20',
'Price_to_Low_20',

# --- Lag features ---
'Return_lag1',
'Return_lag3',
'Vol_lag1',
'Mom_lag1',

'SPY_Return',
'SPY_Volatility',
'VIX_Volatility',
'VIX_Return',

'SMA_200', 
'SMA_50'
]
