import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)
tf.random.set_seed(0)

# -------------------------
# Data (1960-2024; 2024=43 confirmed)
# -------------------------
years = np.arange(1960, 2025)
values = np.array([
    30,45,92,62,15,28,82,16,30,3,
    94,54,89,33,31,19,51,52,50,74,
    44,4,96,15,17,30,40,94,59,7,
    37,50,25,91,75,39,8,65,72,6,
    9,83,33,20,62,3,40,61,66,90,
    41,25,70,67,29,63,93,6,19,15,
    66,19,0,94,43  # includes 2024
])
assert len(years) == len(values)

# Utility: clamp predictions to 0-99 integer
def clamp_pred(x):
    x = int(round(x))
    return x % 100

# -------------------------
# Quick exploratory plots
# -------------------------
plt.figure(figsize=(10,3))
plt.plot(years, values, marker='o', linewidth=1)
plt.title("Sequence 1960-2024")
plt.xlabel("Year")
plt.ylabel("Value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------
# Backtesting helper (rolling-origin)
# -------------------------
def rolling_forecast(model_fit_func, window_train=40, step=1, verbose=False):
    """
    model_fit_func(train_years, train_values) -> prediction_for_next_year (float)
    Runs walk-forward: for each origin starting at (start_index= len- window_train - 1)
    returns arrays of (y_true, y_pred)
    """
    n = len(values)
    y_true = []
    y_pred = []
    # Start so we have enough training history (window_train years)
    start = window_train
    for t in range(start, n-1):  # predict t+1 using data[:t+1]
        train_y = values[:t+1]
        pred = model_fit_func(train_y)
        y_true.append(values[t+1])
        y_pred.append(pred)
        if verbose:
            print(f"Origin {years[t]} -> predict {years[t+1]} : pred={pred}, true={values[t+1]}")
    return np.array(y_true), np.array(y_pred)

# -------------------------
# Model 1: ARIMA (auto order via AIC grid search)
# -------------------------
def fit_arima_and_predict(train_y):
    # train_y: 1D numpy array of integers
    # auto-select small p,d,q by AIC (grid)
    best_aic = np.inf
    best_order = None
    best_model = None
    max_pq = 4
    # Allow d up to 2
    for p in range(0, max_pq+1):
        for d in range(0, 3):
            for q in range(0, max_pq+1):
                try:
                    # statsmodels sometimes fails; catch it
                    mod = ARIMA(train_y, order=(p,d,q))
                    res = mod.fit(method_kwargs={"warn_convergence": False}, disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p,d,q)
                        best_model = res
                except Exception:
                    continue
    # predict next step
    forecast = best_model.forecast(steps=1)
    return float(forecast[0])

# Quick one-shot fit on full data to predict 2025
arima_pred_2025 = fit_arima_and_predict(values)
print("ARIMA raw prediction for 2025:", arima_pred_2025, "->", clamp_pred(arima_pred_2025))

# -------------------------
# Model 2: LSTM (sequence to one)
# -------------------------
def create_lstm_predictor(train_y, window=5, epochs=300, batch_size=8):
    # scale 0-1
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(train_y.reshape(-1,1)).flatten()
    # prepare sequences
    X, Y = [], []
    for i in range(len(scaled)-window):
        X.append(scaled[i:i+window])
        Y.append(scaled[i+window])
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # Build model
    model = Sequential([
        LSTM(64, input_shape=(window,1), activation='tanh'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=0)
    # Fit (quiet)
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    # prepare last window to predict next
    last_seq = scaled[-window:].reshape(1, window, 1)
    pred_scaled = model.predict(last_seq, verbose=0)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    return float(pred)

# LSTM predict
try:
    lstm_pred_2025 = create_lstm_predictor(values, window=5, epochs=500, batch_size=8)
    print("LSTM raw prediction for 2025:", lstm_pred_2025, "->", clamp_pred(lstm_pred_2025))
except Exception as e:
    print("LSTM failed to run here (likely need TensorFlow environment). Error:", e)
    lstm_pred_2025 = None

# -------------------------
# Model 3: Polynomial regression (degrees 1..6) choose best by CV on last 30 years
# -------------------------
def poly_predict(train_y, degree=3):
    # train on full series: X = year index (0..n-1)
    X = np.arange(len(train_y)).reshape(-1,1)
    # create polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    pf = PolynomialFeatures(degree)
    Xpf = pf.fit_transform(X)
    lr = LinearRegression()
    lr.fit(Xpf, train_y)
    next_x = np.array([[len(train_y)]])
    next_xpf = pf.transform(next_x)
    pred = lr.predict(next_xpf)[0]
    return float(pred), lr, pf

# pick degree by simple holdout
best_deg = None
best_mse = np.inf
best_poly_pred = None
for deg in range(1,7):
    # simple train/val on last 30 years
    if len(values) < 40: break
    train_subset = values[:-10]
    pred, lr_model, pf = poly_predict(train_subset, degree=deg)  # predict one step after train_subset
    # compute mse on validation (next 10 points)
    # For robust approach we'd do rolling, but keep lightweight here
    # We'll just evaluate prediction error on the next point as proxy
    # (this is a simplification)
    # use val point (train_subset length)
    val_true = values[len(train_subset)]
    mse = (pred - val_true)**2
    if mse < best_mse:
        best_mse = mse
        best_deg = deg
        best_poly_pred = pred

print("Best poly degree:", best_deg, "poly pred raw:", best_poly_pred, "->", clamp_pred(best_poly_pred))

# -------------------------
# Digit-cycle analysis
# -------------------------
units = values % 10
tens = (values // 10) % 10

def autocorr(x, lag):
    x = np.array(x)
    if lag >= len(x): return np.nan
    return np.corrcoef(x[:-lag], x[lag:])[0,1]

# check autocorrelations for lags 1..10
print("\nUnits-digit autocorr lags 1..10:")
for lag in range(1,11):
    print(f"lag {lag}: {autocorr(units, lag):.3f}", end=" | ")
print("\nTens-digit autocorr lags 1..10:")
for lag in range(1,11):
    print(f"lag {lag}: {autocorr(tens, lag):.3f}", end=" | ")
print()

# check repeating patterns in last N years (search for small periodicities)
def find_periodicities(arr, max_period=20):
    arr = np.array(arr)
    n = len(arr)
    scores = {}
    for p in range(2, max_period+1):
        # measure how often arr[i] == arr[i-p] (mod 100)
        matches = 0
        count = 0
        for i in range(p, n):
            count += 1
            if arr[i] == arr[i-p]:
                matches += 1
        scores[p] = matches / count if count>0 else 0.0
    return scores

period_scores = find_periodicities(values, max_period=30)
# print best candidates
sorted_periods = sorted(period_scores.items(), key=lambda x: -x[1])[:6]
print("\nTop periodicity candidates (period -> match fraction):")
for p,score in sorted_periods:
    print(p, "->", f"{score:.2f}")

# -------------------------
# Modular checks (mod m) for m up to 12
# -------------------------
def mod_pattern_score(arr, m):
    arr = np.array(arr) % m
    # compute autocorrelation of mod series at lag 1
    return autocorr(arr, 1)

print("\nModular autocorr (lag1) for m=2..12:")
for m in range(2,13):
    print(f"mod {m}: {mod_pattern_score(values, m):.3f}", end=" | ")
print()

# -------------------------
# Simple Markov-like transition of last digit
# -------------------------
def transition_matrix(seq, base=100):
    seq = [int(x)%base for x in seq]
    states = sorted(list(set(seq)))
    idx = {s:i for i,s in enumerate(states)}
    M = np.zeros((len(states), len(states)))
    for a,b in zip(seq[:-1], seq[1:]):
        M[idx[a], idx[b]] += 1
    # normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.nan_to_num(M / M.sum(axis=1, keepdims=True))
    return states, M

states, M = transition_matrix(values, base=100)
# find most likely next state given last value
last_val = int(values[-1])
if last_val in states:
    last_idx = states.index(last_val)
    probs = M[last_idx]
    if probs.sum() > 0:
        next_state = states[np.argmax(probs)]
        print("\nMarkov transition predicts next value (most likely):", next_state)
    else:
        print("\nMarkov matrix has no outgoing transitions from last state.")
else:
    print("\nLast value not in Markov state list (shouldn't happen).")

# -------------------------
# Hybrid ensemble
#   - We'll compute simple weights from backtest MAE (lower MAE -> higher weight)
# -------------------------
# For weight estimation, define quick wrapper functions:
def arima_wrapper(train_y):
    return fit_arima_and_predict(train_y)

def poly_wrapper(train_y):
    pred,_,_ = poly_predict(train_y, degree=best_deg if best_deg is not None else 3)
    return pred

def lstm_wrapper(train_y):
    try:
        return create_lstm_predictor(train_y, window=5, epochs=200, batch_size=8)
    except Exception:
        # if LSTM can't run in env, fallback to mean
        return float(np.mean(train_y))

# compute rolling forecast errors for each model
print("\nRunning rolling backtest (this may take a little while)...")
y_true_arima, y_pred_arima = rolling_forecast(lambda t: arima_wrapper(t), window_train=35, step=1, verbose=False)
y_true_poly, y_pred_poly = rolling_forecast(lambda t: poly_wrapper(t), window_train=35, step=1, verbose=False)
# LSTM rolling is expensive; do smaller window
y_true_lstm, y_pred_lstm = rolling_forecast(lambda t: lstm_wrapper(t), window_train=45, step=1, verbose=False)

def score(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), sqrt(mean_squared_error(y_true, y_pred))

mae_arima, rmse_arima = score(y_true_arima, y_pred_arima)
mae_poly, rmse_poly = score(y_true_poly, y_pred_poly)
mae_lstm, rmse_lstm = score(y_true_lstm, y_pred_lstm)

print(f"Backtest MAE: ARIMA={mae_arima:.2f}, POLY={mae_poly:.2f}, LSTM={mae_lstm:.2f}")

# Build weights proportional to 1/MAE (add epsilon)
eps = 1e-6
w_arima = 1.0 / (mae_arima + eps)
w_poly  = 1.0 / (mae_poly + eps)
w_lstm  = 1.0 / (mae_lstm + eps)
w_sum = w_arima + w_poly + w_lstm
w_arima /= w_sum; w_poly /= w_sum; w_lstm /= w_sum
print("Ensemble weights (ARIMA, POLY, LSTM):", w_arima, w_poly, w_lstm)

# Final predictions using full data as train
p_arima = fit_arima_and_predict(values)
p_poly,_,_ = poly_predict(values, degree=best_deg if best_deg is not None else 3)
try:
    p_lstm = create_lstm_predictor(values, window=5, epochs=400, batch_size=8)
except Exception:
    p_lstm = float(np.mean(values))

ensemble_raw = w_arima*p_arima + w_poly*p_poly + w_lstm*p_lstm
ensemble_pred = clamp_pred(ensemble_raw)
print("\nFinal model raw preds -> ARIMA:", p_arima, "POLY:", p_poly, "LSTM:", p_lstm)
print("Ensembled raw:", ensemble_raw, "-> Ensembeled 2025 prediction (0-99):", ensemble_pred)

# Additional diagnostics: residual plot for ARIMA on full fit
try:
    arima_full = ARIMA(values, order=(1,0,1)).fit()
    resid = arima_full.resid
    sm.graphics.tsa.plot_acf(resid, lags=20, title="ARIMA residual ACF")
    plt.show()
except Exception:
    pass

print("\n--- DONE ---")
print("Notes: If LSTM fails (tensorflow issues), try running in an environment with TF installed and GPU disabled for stability.")
