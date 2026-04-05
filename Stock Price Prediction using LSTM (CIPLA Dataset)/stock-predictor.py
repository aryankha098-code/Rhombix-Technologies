import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & prepare data ──────────────────────────────────────────────────
df = pd.read_csv('CIPLA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df[['Date', 'Close']].dropna()

print(f"Dataset: {len(df)} trading days  |  {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Close price range: ₹{df['Close'].min():.2f} – ₹{df['Close'].max():.2f}")

# ── 2. Scale ─────────────────────────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df[['Close']])

# ── 3. Sequence builder ───────────────────────────────────────────────────────
SEQ_LEN = 60   # look-back window

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled, SEQ_LEN)
X = X.reshape(X.shape[0], X.shape[1], 1)   # (samples, time-steps, features)

# ── 4. Train / test split (80 / 20) ─────────────────────────────────────────
split = int(len(X) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = df['Date'].values[SEQ_LEN + split:]

print(f"\nTrain samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# ── 5. Build LSTM model ───────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ── 6. Train ──────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ── 7. Predict ────────────────────────────────────────────────────────────────
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# ── 8. Metrics ────────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

print(f"\n{'─'*40}")
print(f"MAE  : ₹{mae:.2f}")
print(f"RMSE : ₹{rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"{'─'*40}")

# ── 9. Future 30-day forecast ─────────────────────────────────────────────────
last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
future_preds = []
seq = last_seq.copy()

for _ in range(30):
    p = model.predict(seq, verbose=0)[0, 0]
    future_preds.append(p)
    seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)

future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
last_date = df['Date'].max()
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)

print("\n30-Day Forecast (next 5 shown):")
for d, p in zip(future_dates[:5], future_prices[:5]):
    print(f"  {d.date()}  ₹{p:.2f}")
print(f"  …")
print(f"  {future_dates[-1].date()}  ₹{future_prices[-1]:.2f}  (day 30)")

# ── 10. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 16))
fig.patch.set_facecolor('#0f1117')
for ax in axes:
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#c0c0c0')
    ax.xaxis.label.set_color('#c0c0c0')
    ax.yaxis.label.set_color('#c0c0c0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

CIPLA_BLUE  = '#00b4d8'
CIPLA_GREEN = '#06d6a0'
CIPLA_RED   = '#ef476f'
CIPLA_GOLD  = '#ffd166'

# ── Panel 1: Full history + forecast ─────────────────────────────────────────
ax1 = axes[0]
ax1.plot(df['Date'], df['Close'], color=CIPLA_BLUE, linewidth=1, alpha=0.8, label='Historical Close')
ax1.plot(future_dates, future_prices, color=CIPLA_GOLD, linewidth=2.5,
         linestyle='--', label='30-Day Forecast', zorder=5)
ax1.axvline(df['Date'].max(), color='#ffffff', linestyle=':', linewidth=1, alpha=0.4)
ax1.fill_between(future_dates, future_prices * 0.95, future_prices * 1.05,
                 alpha=0.15, color=CIPLA_GOLD, label='±5% Band')
ax1.set_title('CIPLA — Full Price History + 30-Day LSTM Forecast',
              color='white', fontsize=14, fontweight='bold', pad=10)
ax1.set_ylabel('Price (₹)', color='#c0c0c0')
ax1.legend(framealpha=0.2, labelcolor='white', fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator(3))

# ── Panel 2: Test-set actual vs predicted ────────────────────────────────────
ax2 = axes[1]
test_dates = pd.to_datetime(dates_test)
ax2.plot(test_dates, y_actual, color=CIPLA_BLUE,  linewidth=1.5, label='Actual')
ax2.plot(test_dates, y_pred,   color=CIPLA_GREEN, linewidth=1.5, linestyle='--', label='Predicted')
ax2.set_title('Test Set: Actual vs LSTM Predicted',
              color='white', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel('Price (₹)', color='#c0c0c0')
ax2.legend(framealpha=0.2, labelcolor='white', fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

# annotation box
props = dict(boxstyle='round', facecolor='#1a1d27', edgecolor='#555', alpha=0.8)
ax2.text(0.02, 0.96,
         f"MAE: ₹{mae:.2f}  |  RMSE: ₹{rmse:.2f}  |  MAPE: {mape:.2f}%",
         transform=ax2.transAxes, fontsize=9, color='white',
         verticalalignment='top', bbox=props)

# ── Panel 3: Training loss curve ──────────────────────────────────────────────
ax3 = axes[2]
ax3.plot(history.history['loss'],     color=CIPLA_BLUE,  linewidth=2,   label='Train Loss')
ax3.plot(history.history['val_loss'], color=CIPLA_RED,   linewidth=2, linestyle='--', label='Val Loss')
ax3.set_title('Training & Validation Loss', color='white', fontsize=13, fontweight='bold', pad=10)
ax3.set_xlabel('Epoch', color='#c0c0c0')
ax3.set_ylabel('MSE Loss', color='#c0c0c0')
ax3.legend(framealpha=0.2, labelcolor='white', fontsize=9)

plt.tight_layout(pad=2.5)
out_path = 'cipla_lstm_forecast.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nPlot saved → {out_path}")