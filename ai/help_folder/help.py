from sklearn.preprocessing import StandardScaler
import yfinance as yf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
import pandas as pd 
import os
import joblib
import json
from ai.features import FEATURES

def download(ticker,start,end,interval="1d"):
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

    df = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        df['Open'] = data['Open'].iloc[:, 0]
        df['High'] = data['High'].iloc[:, 0]
        df['Low'] = data['Low'].iloc[:, 0]
        df['Close'] = data['Close'].iloc[:, 0]
        df['Volume'] = data['Volume'].iloc[:, 0]
    else:
        df['Open'] = data['Open']
        df['High'] = data['High']
        df['Low'] = data['Low']
        df['Close'] = data['Close']
        df['Volume'] = data['Volume']
    df = df.dropna().reset_index(drop=True)
    return data, df
def calculate_base_features(df, spy=None, vix=None):
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        known = {"open", "high", "low", "close", "volume", "adj_close", "adjclose"}
        flattened = []
        for col in data.columns:
            parts = [str(x) for x in col]
            chosen = None
            for p in reversed(parts):
                norm = p.strip().lower().replace(" ", "_")
                if norm in known:
                    chosen = p
                    break
            flattened.append(chosen if chosen is not None else parts[-1])
        data.columns = flattened
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    close_candidates = ['close', 'adj_close', 'adjclose']
    selected_close_col = None
    
    for col in close_candidates:
        if col in data.columns:
            selected_close_col = col
            break

    if selected_close_col is None:
        raise KeyError("❌ Не найдена колонка 'Close' или 'Adj Close'")

    close_series = data[selected_close_col]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]  
    
    data['Close'] = close_series.squeeze()
    rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}
    data = data.rename(columns=rename_map)
    required_cols = ['Close', 'High', 'Low', 'Volume']
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"❌ Отсутствуют колонки: {missing}")

    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['SMA_200'] = data['Close'].rolling(200).mean()
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    data['Return'] = data['Close'].pct_change()
    data['Return_5d'] = data['Close'].pct_change(5)
    data['Return_10d'] = data['Close'].pct_change(10)
    data['Momentum'] = data['Close'] - data['Close'].shift(5)
    data['ROC_10'] = data['Close'].pct_change(10)
    data['Vol_20d'] = data['Return'].rolling(20).std() * np.sqrt(252)
    median_vol = data['Vol_20d'].rolling(252).median()
    data['Volatility_Ratio'] = data['Vol_20d'] / (median_vol + 1e-9)
    # RSI
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    data['MACD'] = macd_line
    data['MACD_Signal'] = signal_line
    data['MACD_Hist'] = macd_line - signal_line
    # Bollinger Bands & Ratios
    data['BB_Ratio'] = (data['Close'] - sma20 + 1e-9) / (2 * std20 + 1e-9)
    data['Trend_50_Ratio'] = data['Close'] / (data['SMA_50'] + 1e-9)
    data['Trend_200_Ratio'] = data['Close'] / (data['SMA_200'] + 1e-9)
    data['ZScore_20'] = (data['Close'] - sma20) / (std20 + 1e-9)
    data['Price_to_High_20'] = data['Close'] / data['High'].rolling(20).max()
    data['Price_to_Low_20'] = data['Close'] / data['Low'].rolling(20).min()
    # Lag features
    data['Return_lag1'] = data['Return'].shift(1)
    data['Return_lag3'] = data['Return'].shift(3)
    data['Vol_lag1'] = data['Vol_20d'].shift(1)
    data['Mom_lag1'] = data['Momentum'].shift(1)
    # Trend Regime
    data['Trend_Regime'] = 0
    data.loc[data['Close'] > data['SMA_50'], 'Trend_Regime'] = 1

    if spy is not None:
        spy_data = spy.copy()
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = ['_'.join(map(str, col)).strip().lower().replace(' ', '_') for col in spy_data.columns]
        else:
            spy_data.columns = spy_data.columns.str.strip().str.lower().str.replace(' ', '_')

        close_candidates = [c for c in spy_data.columns if 'close' in c]
        if close_candidates:
            close_col = close_candidates[0]
            spy_close = spy_data[close_col]
            if isinstance(spy_close, pd.DataFrame):
                spy_close = spy_close.iloc[:, 0]
            spy_data['SPY_Return'] = spy_close.pct_change()
            spy_data['SPY_Volatility'] = spy_data['SPY_Return'].rolling(20).std() * np.sqrt(252)
        else:
            spy_data['SPY_Return'] = 0.0
            spy_data['SPY_Volatility'] = 0.0

        data['SPY_Return'] = spy_data['SPY_Return'].reindex(data.index).ffill().fillna(0.0)
        data['SPY_Volatility'] = spy_data['SPY_Volatility'].reindex(data.index).ffill().fillna(0.0)
    else:
        data['SPY_Return'] = 0.0
        data['SPY_Volatility'] = 0.0

    if vix is not None:
        vix_data = vix.copy()
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = ['_'.join(map(str, col)).strip().lower().replace(' ', '_') for col in vix_data.columns]
        else:
            vix_data.columns = vix_data.columns.str.strip().str.lower().str.replace(' ', '_')

        vix_close_candidates = [c for c in vix_data.columns if 'close' in c]
        if vix_close_candidates:
            close_col = vix_close_candidates[0]
            vix_close = vix_data[close_col]
            if isinstance(vix_close, pd.DataFrame):
                vix_close = vix_close.iloc[:, 0]
            
            vix_data['VIX_Return'] = vix_close.pct_change()
            vix_data['VIX_Volatility'] = vix_data['VIX_Return'].rolling(20).std() * np.sqrt(252)
        else:
            vix_data['VIX_Return'] = 0.0
            vix_data['VIX_Volatility'] = 0.0

        data['VIX_Return'] = vix_data['VIX_Return'].reindex(data.index).ffill().fillna(0.0)
        data['VIX_Volatility'] = vix_data['VIX_Volatility'].reindex(data.index).ffill().fillna(0.0)
    else:
        data['VIX_Return'] = 0.0
        data['VIX_Volatility'] = 0.0

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0.0)

    print(f"✅ Создано {len(data.columns)} признаков. Всего строк: {len(data)}")
    return data

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return super(MyEncoder, self).default(obj)
    
def save_artifacts_json(
    base_dir,
    strategy_name,
    res_stage1,
    res_stage2,
    FEATURES,
    params,
    backtest_results
):
    folder = os.path.join(base_dir, strategy_name)
    os.makedirs(folder, exist_ok=True)

    artifacts = {
        "strategy_name": strategy_name,
        "params": params,
        "features": FEATURES,
        "stage1": {
            "threshold": res_stage1.get('threshold'),
            "f1_val": res_stage1.get('f1_val'),
            "n_signals": res_stage1.get('n_signals_test'),
            "class_balance": res_stage1.get('class_balance', None)
        },
        "stage2": {
            "threshold": res_stage2.get('threshold'),
            "f1_val": res_stage2.get('f1_val'),
            "n_signals": len(res_stage2.get('y_true', [])),
            "class_balance": res_stage2.get('class_balance', None)
        },
        "backtest": {
            "total_return": backtest_results.get('total_return'),
            "sharpe_ratio": backtest_results.get('sharpe_ratio'),
            "max_drawdown": backtest_results.get('max_drawdown'),
            "win_rate": backtest_results.get('win_rate'),
            "trades": backtest_results.get('trades'),
            "avg_pnl": backtest_results.get('avg_pnl')
        }
    }
    json_path = os.path.join(folder, "artifacts_summary.json")
    with open(json_path, 'w') as f:
        json.dump(artifacts, f, indent=4, cls=MyEncoder)

    print(f"✅ JSON summary saved: {json_path}")
    return json_path
def save_models(base_dir, model_stage1, model_stage2, scaler, features, params, strategy_name="strategy"):
    folder = os.path.join(base_dir, strategy_name)
    os.makedirs(folder, exist_ok=True)
    
    model1_path = os.path.join(folder, "model_stage1.h5")
    model2_path = os.path.join(folder, "model_stage2.h5")
    scaler_path = os.path.join(folder, "scaler.pkl")
    features_path = os.path.join(folder, "features.pkl")
    params_path = os.path.join(folder, "params.pkl")
    params_to_save = params.copy()
    params_to_save['features'] = features 

    if model_stage1 is not None:
        model_stage1.save(model1_path)
        print(f"✅ Model Stage 1 saved: {model1_path}")
    else:
        print("⚠️ Model Stage 1 is None, skipping...")
    
    if model_stage2 is not None:
        model_stage2.save(model2_path)
        print(f"✅ Model Stage 2 saved: {model2_path}")
    else:
        print("⚠️ Model Stage 2 is None (no training data), skipping...")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(features, features_path)
    joblib.dump(params_to_save, params_path)
    
    print(f"✅ Scaler, features, and params (WITH features list) saved to {folder}")
    print(f"\n📁 Все артефакты сохранены в {folder}")
def generate_final_report(
        results,
        model_type="single",  
        ml_metrics=None,
        stage1_metrics=None,
        stage2_metrics=None,
        feature_df=None,
        extra_info=None):

    print("\n" + "="*70)
    print("📋 ИТОГОВЫЙ ОТЧЁТ")

    if model_type == "two_stage" and stage1_metrics and stage2_metrics:
        print(f"""
🎯 Two-Stage Classification:
   Stage 1: MOVE vs HOLD (Threshold={stage1_metrics.get('threshold','-')}, F1={stage1_metrics.get('f1','-'):.4f})
   Stage 2: UP vs DOWN (Threshold={stage2_metrics.get('threshold','-')}, F1={stage2_metrics.get('f1','-'):.4f})
""")

    if ml_metrics:
        print("Метрики:")
        for k, v in ml_metrics.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")
        print()

    if results:
        print("💰 Production Backtest:")
        if "total_return" in results:
            print(f"   Total Return: {results['total_return']:.1%}")
        if "sharpe" in results:
            print(f"   Sharpe Ratio: {results['sharpe']:.2f}")
        if "sortino" in results:
            print(f"   Sortino Ratio: {results['sortino']:.2f}")
        if "calmar" in results:
            print(f"   Calmar Ratio: {results['calmar']:.2f}")
        if "max_drawdown" in results:
            print(f"   Max Drawdown: {results['max_drawdown']:.1%}")
        if "trades" in results or "num_trades" in results:
            print(f"Trades: {len(results['trades'])}")
        if "win_rate" in results:
            print(f"   Win Rate: {results['win_rate']:.1%}")
        if "filtered_ratio" in results:
            print(f"   Filtered Signals: {results['filtered_ratio']:.1%}")

    if feature_df is not None:
        for i in range(min(5, len(feature_df))):
            print(f"   {i+1}. {feature_df['Feature'].iloc[i]}")

    if extra_info:
        print("\n⚙ Дополнительная информация:")
        for k, v in extra_info.items():
            print(f"   {k}: {v}")
    print("="*70)
 
def create_stage1_labels(close, high, low, thresholds, horizon=5, mask_move=None):

    labels = []
    n = len(close)

    for i in range(n - horizon):

        if np.isnan(thresholds[i]):
            labels.append(0)
            continue

        if mask_move is not None and mask_move[i] != 1:
            labels.append(0)
            continue

        start_price = close[i]

        future_high = high[i+1:i+horizon+1]
        future_low  = low[i+1:i+horizon+1]

        up_move = (np.max(future_high) - start_price) / start_price
        down_move = (start_price - np.min(future_low)) / start_price

        if up_move >= thresholds[i] or down_move >= thresholds[i]:
            labels.append(1)
        else:
            labels.append(0)

    # добавляем хвост
    labels += [0] * horizon

    return np.array(labels)
def create_stage2_labels(close, horizon=5,mask_move=None):
    """
    Stage 2: UP (1) vs DOWN (0)
    """
    n = len(close)
    labels = []
    
    for i in range(n - horizon):
        change = (close[i + horizon] - close[i]) / close[i]
        labels.append(1 if change > 0 else 0)
    
    for _ in range(horizon):
        labels.append(np.nan)
    return np.array(labels)

def prepare_data(
    df,
    train_split,
    val_split,
    seq_length,
    move_threshold=0.03,
    horizon=5,
    FEATURES1=None):

    returns = np.log(df['Close']).diff()

    volatility = (returns
        .rolling(20)
        .std()
        .bfill()
        .fillna(0)
    )

    dynamic_threshold = (volatility * 2.5).to_numpy()

    assert len(dynamic_threshold) == len(df)
    df['Label_Stage1'] = create_stage1_labels(
        df['Close'].values,
        df['High'].values,
        df['Low'].values,
        thresholds=dynamic_threshold,
        horizon=horizon
    )

    df['Label_Stage2'] = create_stage2_labels(
        df['Close'].values,
        horizon=horizon
    )

    df = df.dropna(subset=['Label_Stage1', 'Label_Stage2']).reset_index(drop=True)

    df['Label_Stage1'] = df['Label_Stage1'].astype(int)
    df['Label_Stage2'] = df['Label_Stage2'].astype(int)

    df['dynamic_threshold'] = dynamic_threshold[-len(df):]

    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    scaler = StandardScaler()
    scaler.fit(df_train[FEATURES1])

    train_scaled = scaler.transform(df_train[FEATURES1])
    val_scaled   = scaler.transform(df_val[FEATURES1])
    test_scaled  = scaler.transform(df_test[FEATURES1])

    def create_sequences(data, labels, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(labels[i + seq_len])
        return np.array(X), np.array(y)

    X_train, y1_train = create_sequences(train_scaled, df_train['Label_Stage1'].values, seq_length)
    X_val,   y1_val   = create_sequences(val_scaled,   df_val['Label_Stage1'].values,   seq_length)
    X_test,  y1_test  = create_sequences(test_scaled,  df_test['Label_Stage1'].values,  seq_length)

    _, y2_train = create_sequences(train_scaled, df_train['Label_Stage2'].values, seq_length)
    _, y2_val   = create_sequences(val_scaled,   df_val['Label_Stage2'].values,   seq_length)
    _, y2_test  = create_sequences(test_scaled,  df_test['Label_Stage2'].values,  seq_length)

    df_test_seq = df_test.iloc[seq_length:].reset_index(drop=True)

    assert len(X_test) == len(y1_test) == len(y2_test)
    assert len(df_test_seq) == len(X_test)

    print(f"✅ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    def get_class_weights(y):
        classes = np.unique(y)
        if len(classes) < 2:
            return {0: 1, 1: 1}
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))

    cw1 = get_class_weights(y1_train.astype(int))
    cw2 = get_class_weights(y2_train.astype(int))

    mask1_train = (y1_train == 1)
    mask1_val   = (y1_val == 1)
    mask1_test  = (y1_test == 1)

    df_test_seq['dynamic_threshold'] = dynamic_threshold[-len(df_test_seq):]

    if 'dynamic_threshold' in df_test_seq.columns:
        pass
    else:
        df_test_seq['dynamic_threshold'] = dynamic_threshold[-len(df_test_seq):]
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "y1_train": y1_train,
        "y1_val": y1_val,
        "y1_test": y1_test,

        "y2_train": y2_train,
        "y2_val": y2_val,
        "y2_test": y2_test,

        "mask1_train": mask1_train,
        "mask1_val": mask1_val,
        "mask1_test": mask1_test,

        "df_test": df_test_seq,

        "scaler": scaler,
        "cw1": cw1,
        "cw2": cw2
    }
def optimize_threshold(model, X, y, metric='precision', low=0.45, high=0.75, step=0.01, min_signal_ratio=0.02):
    print(f"\n🔍 Оптимизация порога (metric='{metric}', min_signals={min_signal_ratio*100:.0f}%)...")
    probs = model.predict(X, verbose=0).flatten()
    n_samples = len(y)
    
    thresholds = np.arange(low, high + step, step)
    best_threshold = 0.5
    best_score = -np.inf
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        signal_ratio = preds.sum() / n_samples
        
        if signal_ratio < min_signal_ratio:
            continue
            
        if metric == 'precision':
            score = precision_score(y, preds, zero_division=0)
        elif metric == 'f1':
            score = f1_score(y, preds, zero_division=0)
        elif metric == 'accuracy':
            score = np.mean(preds == y)
        else:
            raise ValueError(f"Unknown metric {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    print(f"Лучший порог: {best_threshold:.2f}")
    print(f"Сигналов: {(probs >= best_threshold).sum()} ({100*(probs >= best_threshold).mean():.1f}%)")
    return best_threshold, probs, best_score


def stage1(
    X_train, X_val, X_test,
    y1_train, y1_val, y1_test,
    cw1,
    df_test,
    create_model,
    optimize_threshold,
    filter_signals=None,
    EPOCHS=100,
    BATCH_SIZE=32,
    PATIENCE_ES=10,
    PATIENCE_LR=5):

    print("\n STAGE 1: MOVE vs HOLD")
    print("=" * 50)

    input_shape = X_train.shape[1:]
    model = create_model(input_shape, name='stage1_move')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE_ES,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=PATIENCE_LR,
        min_lr=1e-6,
        verbose=1
    )

    history1 = model.fit(
        X_train, y1_train,
        validation_data=(X_val, y1_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    probs_train = model.predict(X_train, verbose=0).flatten()
    probs_val   = model.predict(X_val, verbose=0).flatten()
    probs_test  = model.predict(X_test, verbose=0).flatten()

    threshold, _, f1 = optimize_threshold(model, X_val, y1_val, metric='precision')
    print(f"\n Threshold (val): {threshold:.3f}, F1: {f1:.4f}")

    preds_train = (probs_train >= threshold).astype(int)
    preds_val   = (probs_val   >= threshold).astype(int)
    preds_test  = (probs_test  >= threshold).astype(int)

    mask_train = (preds_train == 1)
    mask_val   = (preds_val   == 1)
    mask_test  = (preds_test  == 1)

    print(f" MOVE filtered (test): {mask_test.sum()} / {len(mask_test)}")

    regime_features = {
        'regime_score': df_test['Trend_Regime'].values,
        'vol_regime': (df_test['Vol_20d'] > df_test['Vol_20d'].median()).astype(int).values
    }

    filtered_preds, _ = filter_signals(
        preds_test,
        probs_test,
        regime_features,
        PROB_THRESHOLD_BASE=threshold,
    )

    mask_test = (filtered_preds != 0)

    if mask_test.sum() == 0:
        print("WARNING: нет сигналов на тесте")
    if mask_test.sum() == len(mask_test):
        print("WARNING: сигналы везде на тесте")

    return {
        "mask_train": mask_train,
        "mask_val": mask_val,
        "mask_test": mask_test,
        "probs_train": probs_train,
        "probs_val": probs_val,
        "probs_test": probs_test,
        "preds_train": preds_train,
        "preds_val": preds_val,
        "preds_test": preds_test,
        "filtered_preds_test": filtered_preds,
        "threshold": threshold,
        "model": model,
        "f1_val": f1,
        "n_signals_test": int(mask_test.sum()),
        "history": history1
    }
def stage2(
    X_train, X_val, X_test,
    y2_train, y2_val, y2_test,
    mask1_train, mask1_val, mask1_test,
    create_model,
    optimize_threshold,
    EPOCHS=50,
    BATCH_SIZE=32,
    PATIENCE_ES=5,
    PATIENCE_LR=5,
    PROB_THRESHOLD_HIGH=0.75,
    PROB_THRESHOLD_LOW=0.5
    ):

    print("\n STAGE 2: UP vs DOWN")
    print("=" * 50)
    print(f"Stage1 signals → train: {mask1_train.sum()}, val: {mask1_val.sum()}, test: {mask1_test.sum()}")

    X_train_s2, y_train_s2 = X_train[mask1_train], y2_train[mask1_train]
    X_val_s2, y_val_s2 = X_val[mask1_val], y2_val[mask1_val]
    X_test_s2, y_test_s2 = X_test[mask1_test], y2_test[mask1_test]

    train_mask = ~np.isnan(y_train_s2)
    val_mask   = ~np.isnan(y_val_s2)
    test_mask  = ~np.isnan(y_test_s2)

    X_train_clean, y_train_clean = X_train_s2[train_mask], y_train_s2[train_mask].astype(int)
    X_val_clean, y_val_clean     = X_val_s2[val_mask], y_val_s2[val_mask].astype(int)
    X_test_clean, y_test_clean   = X_test_s2[test_mask], y_test_s2[test_mask].astype(int)

    if len(X_train_clean) == 0:
        print("❌ Stage2: нет train данных")
        return empty_stage2()
    if len(X_val_clean) == 0:
        print("❌ Stage2: нет val данных")
        return empty_stage2()
    if len(np.unique(y_train_clean)) < 2:
        print(f"❌ Stage2: один класс {np.unique(y_train_clean)}")
        return empty_stage2()
    
    classes = np.unique(y_train_clean)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_clean)
    cw2 = dict(zip(classes, weights))
    print(f"Class weights: {cw2}")

    model = create_model((X_train_clean.shape[1], X_train_clean.shape[2]))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_LR, verbose=0)
    ]

    history = model.fit(
        X_train_clean, y_train_clean,
        validation_data=(X_val_clean, y_val_clean),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw2,
        callbacks=callbacks,
        verbose=1
    )

    threshold, _, f1_val = optimize_threshold(model, X_val_clean, y_val_clean, metric='f1')
    if len(X_test_clean) == 0:
        print("⚠️ Stage2: нет test данных")
        return empty_stage2()
    
    probs_test = model.predict(X_test_clean, verbose=0).flatten()
    preds_test = np.zeros(len(probs_test))
    # Попробуй так для теста:
    threshold_high = threshold if threshold > 0.5 else 0.51
    threshold_low = 1 - threshold_high  

    # preds_test[probs_test >= PROB_THRESHOLD_HIGH] = 1  # LONG
    # preds_test[probs_test <= PROB_THRESHOLD_LOW] = -1  # SHORT
    preds_test[probs_test >= threshold_high] = 1  # LONG
    preds_test[probs_test <= threshold_low] = -1   # SHORT

    indices_test = np.flatnonzero(mask1_test)
    indices_test = indices_test[~np.isnan(y_test_s2)]    
    assert len(preds_test) == len(y_test_clean) == len(indices_test)

    print(f"📊 Test samples: {len(preds_test)}")
    print(f"UP (Long): {(preds_test == 1).sum()} | DOWN (Short): {(preds_test == -1).sum()} | Wait: {(preds_test == 0).sum()}")

    return {
        "preds": preds_test,
        "probs": probs_test,
        "y_true": y_test_clean,
        "indices": indices_test,
        "threshold": threshold,
        "model": model,
        "f1_val": f1_val,
        "history": history
    }
def empty_stage2():

    return {
        "preds": np.array([]),
        "probs": np.array([]),
        "y_true": np.array([]),
        "indices": np.array([]),
        "threshold": 0.3,
        "model": None,
        "f1_val": 0,
        "history": None
    }

def print_stage_report(y_true, y_pred, title, target_names):
    print(f"\n{title}:")
    
    if len(y_true) > 0:
        print(classification_report(
            y_true, 
            y_pred, 
            labels=[0, 1], 
            target_names=target_names, 
            zero_division=0
        ))
    else:
        print(f"{title}: нет данных для оценки")