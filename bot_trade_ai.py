import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# =============================================================
#   CONFIG GLOBALE
# =============================================================
torch.set_num_threads(1)
gc.enable()

SYMBOL            = "XAUUSD"
TIMEFRAME_M1      = mt5.TIMEFRAME_M1
TIMEFRAME_M5      = mt5.TIMEFRAME_M5
TIMEFRAME_M15     = mt5.TIMEFRAME_M15
TIMEFRAME_H1      = mt5.TIMEFRAME_H1
MAGIC_NUMBER      = 999
DAILY_LOSS_LIMIT  = 0.03   # 3% max par jour
MAX_POSITIONS     = 1
RISK_MONEY        = 8.0    # $ risqués par trade (fixe)
DEVICE            = "cpu"

# Fenêtres modèles
WINDOW_SIZE_M15 = 96
WINDOW_SIZE_M5  = 60
WINDOW_SIZE_M1  = 120

# ── Seuils optimisés conjointement (backtest unifié) ─────────
LONG_T_m15  = 0.50
SHORT_T_m15 = 0.54
LONG_T_m5   = 0.70
SHORT_T_m5  = 0.65
LONG_T_m1   = 0.65   # M1 BUY
SHORT_T_m1  = 0.54   # M1 SELL

# Fenêtres filtre SMA
WINDOW_SMA_M15 = 48   # ~12h de bougies M15
WINDOW_SMA_H1  = 50   # ~50 bougies H1

# =============================================================
#   CLASSES MODÈLES (inchangées)
# =============================================================
class TCNBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x[:, :, :-self.conv.padding[0]]
        x = self.relu(x)
        x = self.norm(x)
        return x.transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, query, key):
        out, _ = self.attn(query, key, key)
        return self.norm(out + query)

class RegimeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
    def forward(self, x):
        return self.fc(x)

class MultiTFModel(nn.Module):
    """Modèle M15 — 2 classes (BUY / SELL)"""
    def __init__(self, dim_m5, dim_m15, dim_h1, dim_h4, regime_dim, hidden=64):
        super().__init__()
        self.emb_m5  = nn.Linear(dim_m5, hidden);  self.emb_m15 = nn.Linear(dim_m15, hidden)
        self.emb_h1  = nn.Linear(dim_h1, hidden);  self.emb_h4  = nn.Linear(dim_h4, hidden)
        self.tcn_m5  = TCNBlock(hidden);  self.tcn_m15 = TCNBlock(hidden)
        self.tcn_h1  = TCNBlock(hidden);  self.tcn_h4  = TCNBlock(hidden)
        self.tr_m5   = TransformerBlock(hidden);  self.tr_m15 = TransformerBlock(hidden)
        self.tr_h1   = TransformerBlock(hidden);  self.tr_h4  = TransformerBlock(hidden)
        self.cross_m15 = CrossAttentionFusion(hidden)
        self.cross_h1  = CrossAttentionFusion(hidden)
        self.cross_h4  = CrossAttentionFusion(hidden)
        self.regime_encoder = RegimeEncoder(regime_dim, hidden_dim=32)
        self.fc = nn.Sequential(
            nn.Linear(hidden + 32, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 2))
    def forward(self, x_m5, x_m15, x_h1, x_h4, regime):
        m5 = self.tr_m5(self.tcn_m5(self.emb_m5(x_m5)))
        m5 = self.cross_m15(m5, self.tr_m15(self.tcn_m15(self.emb_m15(x_m15))))
        m5 = self.cross_h1(m5,  self.tr_h1(self.tcn_h1(self.emb_h1(x_h1))))
        m5 = self.cross_h4(m5,  self.tr_h4(self.tcn_h4(self.emb_h4(x_h4))))
        return self.fc(torch.cat([m5[:, -1, :], self.regime_encoder(regime)], dim=1))

class MultiTFModel2(nn.Module):
    """Modèles M5 et M1 — 3 classes (NEUTRE / BUY / SELL)"""
    def __init__(self, dim_m5, dim_m15, dim_h1, dim_h4, regime_dim, hidden=64):
        super().__init__()
        self.emb_m5  = nn.Linear(dim_m5, hidden);  self.emb_m15 = nn.Linear(dim_m15, hidden)
        self.emb_h1  = nn.Linear(dim_h1, hidden);  self.emb_h4  = nn.Linear(dim_h4, hidden)
        self.tcn_m5  = TCNBlock(hidden);  self.tcn_m15 = TCNBlock(hidden)
        self.tcn_h1  = TCNBlock(hidden);  self.tcn_h4  = TCNBlock(hidden)
        self.tr_m5   = TransformerBlock(hidden);  self.tr_m15 = TransformerBlock(hidden)
        self.tr_h1   = TransformerBlock(hidden);  self.tr_h4  = TransformerBlock(hidden)
        self.cross_m15 = CrossAttentionFusion(hidden)
        self.cross_h1  = CrossAttentionFusion(hidden)
        self.cross_h4  = CrossAttentionFusion(hidden)
        self.regime_encoder = RegimeEncoder(regime_dim, hidden_dim=32)
        self.fc = nn.Sequential(
            nn.Linear(hidden + 32, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 3))
    def forward(self, x_m5, x_m15, x_h1, x_h4, regime):
        m5 = self.tr_m5(self.tcn_m5(self.emb_m5(x_m5)))
        m5 = self.cross_m15(m5, self.tr_m15(self.tcn_m15(self.emb_m15(x_m15))))
        m5 = self.cross_h1(m5,  self.tr_h1(self.tcn_h1(self.emb_h1(x_h1))))
        m5 = self.cross_h4(m5,  self.tr_h4(self.tcn_h4(self.emb_h4(x_h4))))
        return self.fc(torch.cat([m5[:, -1, :], self.regime_encoder(regime)], dim=1))

# =============================================================
#   CHARGEMENT DES MODÈLES
# =============================================================
def load_model(path, model_class, ckpt_ref, n_out):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m = model_class(
        dim_m5=len(ckpt_ref["cols_m5"]),   dim_m15=len(ckpt_ref["cols_m15"]),
        dim_h1=len(ckpt_ref["cols_h1"]),   dim_h4=len(ckpt_ref["cols_h4"]),
        regime_dim=len(ckpt_ref["cols_regime"]), hidden=64).to(DEVICE)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, ckpt

checkpoint,  model1 = (lambda c: (c, *[None]))( torch.load("Test_GO_M15_up.pth", map_location=DEVICE, weights_only=False))
checkpoint2, model2 = (lambda c: (c, *[None]))( torch.load("Test_GO_M5_up.pth",  map_location=DEVICE, weights_only=False))
checkpoint3, model3 = (lambda c: (c, *[None]))( torch.load("Test_GO_M1_up.pth",  map_location=DEVICE, weights_only=False))

# Chargement propre
checkpoint  = torch.load("Test_GO_M15_up.pth", map_location=DEVICE, weights_only=False)
checkpoint2 = torch.load("Test_GO_M5_up.pth",  map_location=DEVICE, weights_only=False)
checkpoint3 = torch.load("Test_GO_M1_up.pth",  map_location=DEVICE, weights_only=False)

model1 = MultiTFModel(
    dim_m5=len(checkpoint["cols_m5"]),   dim_m15=len(checkpoint["cols_m15"]),
    dim_h1=len(checkpoint["cols_h1"]),   dim_h4=len(checkpoint["cols_h4"]),
    regime_dim=len(checkpoint["cols_regime"]), hidden=64).to(DEVICE)
model1.load_state_dict(checkpoint["state_dict"]);  model1.eval()

model2 = MultiTFModel2(
    dim_m5=len(checkpoint2["cols_m5"]),  dim_m15=len(checkpoint2["cols_m15"]),
    dim_h1=len(checkpoint2["cols_h1"]),  dim_h4=len(checkpoint2["cols_h4"]),
    regime_dim=len(checkpoint2["cols_regime"]), hidden=64).to(DEVICE)
model2.load_state_dict(checkpoint2["state_dict"]); model2.eval()

model3 = MultiTFModel2(
    dim_m5=len(checkpoint3["cols_m5"]),  dim_m15=len(checkpoint3["cols_m15"]),
    dim_h1=len(checkpoint3["cols_h1"]),  dim_h4=len(checkpoint3["cols_h4"]),
    regime_dim=len(checkpoint3["cols_regime"]), hidden=64).to(DEVICE)
model3.load_state_dict(checkpoint3["state_dict"]); model3.eval()

# =============================================================
#   FONCTIONS FEATURES
# =============================================================
def get_m1_from_mt5(n=15000):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_M1, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open","high","low","close"]]
    return df.round(2)

def add_tf_features(df, prefix):
    df = df.copy()
    df[f"{prefix}_ret"]      = np.log(df["close"] / df["close"].shift(1))
    df[f"{prefix}_hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)
    df[f"{prefix}_oc_body"]  = (df["close"] - df["open"]) / df["open"]
    df[f"{prefix}_vol"]      = df[f"{prefix}_ret"].rolling(48).std()
    return df

def add_adx(df, period=14):
    """ADX en numpy pur — pas de conflit d'index pandas."""
    df   = df.copy()
    idx  = df.index
    high = df["high"].values.astype(float)
    low  = df["low"].values.astype(float)
    close= df["close"].values.astype(float)
    plus_dm  = np.diff(high,  prepend=high[0])
    minus_dm = np.abs(np.diff(low, prepend=low[0]))
    plus_dm_f  = np.where((plus_dm > minus_dm)   & (plus_dm  > 0), plus_dm,  0.0)
    minus_dm_f = np.where((minus_dm > plus_dm_f) & (minus_dm > 0), minus_dm, 0.0)
    tr2  = np.abs(high - np.roll(close, 1)); tr2[0] = high[0] - low[0]
    tr3  = np.abs(low  - np.roll(close, 1)); tr3[0] = tr2[0]
    tr   = np.maximum(high - low, np.maximum(tr2, tr3))
    atr_s    = pd.Series(tr, index=idx).rolling(period).mean()
    plus_di  = 100 * pd.Series(plus_dm_f,  index=idx).rolling(period).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm_f, index=idx).rolling(period).mean() / atr_s.replace(0, np.nan)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    df["adx"] = dx.rolling(period).mean()
    return df

def add_intraday_features(df):
    df = df.copy()
    df["ret"]         = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"]    = (df["high"] - df["low"]) / df["close"].shift(1)
    df["oc_body"]     = (df["close"] - df["open"]) / df["open"]
    df["vol_rolling"] = df["ret"].rolling(48).std()
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-df["close"].shift(1)).abs(),
                    (df["low"] -df["close"].shift(1)).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(48).mean()
    h, m, d   = df.index.hour, df.index.minute, df.index.dayofweek
    df["hour_sin"]   = np.sin(2*np.pi*h/24);  df["hour_cos"]   = np.cos(2*np.pi*h/24)
    df["minute_sin"] = np.sin(2*np.pi*m/60);  df["minute_cos"] = np.cos(2*np.pi*m/60)
    df["dow_sin"]    = np.sin(2*np.pi*d/7);   df["dow_cos"]    = np.cos(2*np.pi*d/7)
    df["session_london"] = ((h>=8)&(h<16)).astype(int)
    df["session_ny"]     = ((h>=13)&(h<21)).astype(int)
    df["session_asia"]   = ((h>=0)&(h<8)).astype(int)
    return df

def prepare_live_features_multiTF(ckpt, window, t1, t2, t3, t4, p1, p2, p3, p4):
    df_m1 = get_m1_from_mt5(15000)
    if df_m1 is None or df_m1.empty:
        return None, None, None, None, None, None
    ohlc = {"open":"first","high":"max","low":"min","close":"last"}
    df_base = df_m1.resample(t1).agg(ohlc).dropna()
    df_tf2  = df_m1.resample(t2).agg(ohlc).dropna()
    df_tf3  = df_m1.resample(t3).agg(ohlc).dropna()
    df_tf4  = df_m1.resample(t4).agg(ohlc).dropna()

    df_base = add_adx(add_tf_features(df_base, p1))
    df_tf2  = add_tf_features(df_tf2, p2)
    df_tf3  = add_tf_features(df_tf3, p3)
    df_tf4  = add_tf_features(df_tf4, p4)

    df = df_base.copy()
    df = df.join(df_tf2.filter(like=p2+"_"), how="left").ffill()
    df = df.join(df_tf3.filter(like=p3+"_"), how="left").ffill()
    df = df.join(df_tf4.filter(like=p4+"_"), how="left").ffill()
    df = add_intraday_features(df)

    if len(df) < window:
        return None, None, None, None, None, None

    def sc(X, m, s): return (X - m) / (s + 1e-9)
    X_m5  = sc(df[ckpt["cols_m5"]].values,    ckpt["scaler_m5_mean"],  ckpt["scaler_m5_scale"])
    X_m15 = sc(df[ckpt["cols_m15"]].values,   ckpt["scaler_m15_mean"], ckpt["scaler_m15_scale"])
    X_h1  = sc(df[ckpt["cols_h1"]].values,    ckpt["scaler_h1_mean"],  ckpt["scaler_h1_scale"])
    X_h4  = sc(df[ckpt["cols_h4"]].values,    ckpt["scaler_h4_mean"],  ckpt["scaler_h4_scale"])
    X_reg = sc(df[ckpt["cols_regime"]].values, ckpt["scaler_reg_mean"], ckpt["scaler_reg_scale"])

    def to_t(arr, n): return torch.tensor(arr[-n:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return (df,
            to_t(X_m5, window), to_t(X_m15, window),
            to_t(X_h1, window), to_t(X_h4, window),
            torch.tensor(X_reg[-1], dtype=torch.float32).unsqueeze(0).to(DEVICE))

# =============================================================
#   FILTRE DE TENDANCE SMA (M15 et H1 séparés)
# =============================================================
def get_trend(timeframe, window_bars, sma_period=20, atr_period=14):
    """
    Retourne +1 (haussier), -1 (baissier), 0 (neutre).
    Utilise SMA + slope + threshold ATR, identique au backtest.
    """
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, window_bars)
    if rates is None:
        return 0
    df  = pd.DataFrame(rates)
    sma = df["close"].rolling(sma_period).mean()
    atr = (df["high"] - df["low"]).rolling(atr_period).mean().iloc[-1]
    sma_now  = sma.iloc[-1]
    sma_prev = sma.iloc[-2]
    slope    = sma_now - sma_prev
    price    = df["close"].iloc[-1]
    thr      = atr * 0.3
    if price > sma_now + thr and slope > 0:
        return 1
    if price < sma_now - thr and slope < 0:
        return -1
    return 0

# =============================================================
#   GESTION DU RISQUE
# =============================================================
def get_daily_drawdown():
    account = mt5.account_info()
    if account is None:
        return True, 0.0
    today   = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    history = mt5.history_deals_get(today, datetime.now())
    daily_profit = sum(d.profit + d.commission + d.swap for d in history) if history else 0.0
    limit = account.balance * DAILY_LOSS_LIMIT
    return (daily_profit < -limit), daily_profit

def compute_sl_points_from_atr(df):
    """SL adaptatif basé sur l'ATR, risque monétaire reste fixe via le lot."""
    atr   = df["atr"].iloc[-1]
    point = mt5.symbol_info(SYMBOL).point
    atr_pts = int(atr / point)
    return max(600, min(1500, atr_pts))

def compute_lot_from_risk(sl_points):
    """Lot calculé pour que la perte max = RISK_MONEY quels que soient les points SL."""
    symbol_info = mt5.symbol_info(SYMBOL)
    # Sur XAUUSD : 1 lot = 100 oz, 1 point = 0.01$ → valeur SL = sl_points * 0.01 * 100
    sl_value = sl_points * symbol_info.point * 100
    lot = round((RISK_MONEY / sl_value) / symbol_info.volume_step) * symbol_info.volume_step
    return max(symbol_info.volume_min, min(symbol_info.volume_max, lot))

def compute_rr(df, prob, trend):
    """
    RR dynamique basé sur ADX, volatilité ATR et niveau de confiance.
    CORRECTION : utilise la première colonne _ret disponible (pas hardcodé m15_ret).
    """
    adx       = df["adx"].iloc[-1]
    vol_ratio = df["atr"].iloc[-1] / df["close"].iloc[-1]
    # Trouver la colonne _ret disponible (m1_ret, m5_ret, m15_ret selon le TF)
    ret_col   = next((c for c in df.columns if c.endswith("_ret")), "ret")
    trend_str = df[ret_col].rolling(20, min_periods=1).mean().iloc[-1]

    rr = 2.0
    # Confiance élevée → RR plus ambitieux
    threshold = LONG_T_m1 if trend > 0 else SHORT_T_m1
    if prob >= threshold + 0.05:
        rr += 0.5
    # Tendance forte (ADX)
    if adx > 25:
        rr += 0.3
    # Volatilité dans la plage idéale
    if 0.0007 < vol_ratio < 0.0020:
        rr += 0.2
    # Tendance faible → prudence
    if abs(trend_str) < 0.0004:
        rr -= 0.3
    # Clamp final
    return max(1.5, min(4.0, rr))

# =============================================================
#   GESTION DES ORDRES
# =============================================================
def get_current_positions():
    return mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER) or []

def close_positions_by_magic():
    tick = mt5.symbol_info_tick(SYMBOL)
    for pos in get_current_positions():
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price      = tick.bid           if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL,
            "volume": pos.volume, "type": order_type, "position": pos.ticket,
            "price": price, "deviation": 20, "magic": MAGIC_NUMBER,
            "comment": "BOT_CLOSE", "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })

def execute_trade(direction, sl_points, rr):
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return None
    si    = mt5.symbol_info(SYMBOL)
    price = tick.ask if direction == mt5.ORDER_TYPE_BUY else tick.bid
    lot   = compute_lot_from_risk(sl_points)
    pt    = si.point
    sl    = price - sl_points*pt if direction == mt5.ORDER_TYPE_BUY else price + sl_points*pt
    tp    = price + sl_points*rr*pt if direction == mt5.ORDER_TYPE_BUY else price - sl_points*rr*pt

    res = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL,
        "volume": lot, "type": direction, "price": price,
        "sl": sl, "tp": tp, "magic": MAGIC_NUMBER,
        "comment": "NR² Trading 😱 pro", "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Trade failed: {res.comment}")
    else:
        side = "BUY" if direction == mt5.ORDER_TYPE_BUY else "SELL"
        print(f"{side} | ticket={res.order} | lot={lot:.2f} | sl={sl:.2f} | tp={tp:.2f} | rr={rr:.2f}")
    return res

def update_trailing_stops():
    """
    Trailing stop en points (cohérent avec le SL d'ouverture).
    Phase 1 : breakeven dès +800 pts de profit
    Phase 2 : trailing de 800 pts dès +1500 pts de profit
    """
    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    if not positions:
        return
    tick  = mt5.symbol_info_tick(SYMBOL)
    point = mt5.symbol_info(SYMBOL).point
    trail = 800 * point   # trailing distance en $

    for pos in positions:
        if pos.type == mt5.ORDER_TYPE_BUY:
            profit_pts = (tick.bid - pos.price_open) / point
            if profit_pts >= 800 and pos.sl < pos.price_open:
                # Phase 1 : move SL to breakeven
                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket,
                                "sl": pos.price_open + point, "tp": pos.tp})
            elif profit_pts >= 1500:
                # Phase 2 : trailing
                new_sl = tick.bid - trail
                if new_sl > pos.sl + point:
                    mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket,
                                    "sl": new_sl, "tp": pos.tp})
        elif pos.type == mt5.ORDER_TYPE_SELL:
            profit_pts = (pos.price_open - tick.ask) / point
            if profit_pts >= 800 and pos.sl > pos.price_open:
                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket,
                                "sl": pos.price_open - point, "tp": pos.tp})
            elif profit_pts >= 1500:
                new_sl = tick.ask + trail
                if new_sl < pos.sl - point:
                    mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket,
                                    "sl": new_sl, "tp": pos.tp})

# =============================================================
#   FILTRE VOLATILITÉ
# =============================================================
def vol_filter_ok(df):
    """Retourne True si la volatilité actuelle dépasse le quantile 30%."""
    vol_thr = df["vol_rolling"].quantile(0.30)
    return df["vol_rolling"].iloc[-1] >= vol_thr

# =============================================================
#   BOUCLE PRINCIPALE
# =============================================================


MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
if not mt5.initialize(path=MT5_PATH):
    print("MT5 init failed"); quit()

account = mt5.account_info()
print(f"💰 Compte : {account.balance:.2f}$")
print(f"Bot démarré | SL fixe: {RISK_MONEY}$ | Daily SL: {DAILY_LOSS_LIMIT*100:.1f}%")

# État inter-cycles
check_confirm      = False
check_confirm_time = None          # NOUVEAU : timestamp du check_confirm
current_h1_bias    = 0             # NOUVEAU : vrai filtre H1 séparé
confiance_buy_m15  = confiance_sell_m15 = 0.0
confiance_buy_m5   = confiance_sell_m5  = 0.0
last_h1_update     = None          # pour ne pas recalculer H1 à chaque seconde

try:
    while True:
        now = datetime.now()

        # ── 1. Daily Stop Loss ────────────────────────────────
        stopped, day_pnl = get_daily_drawdown()
        if stopped:
            print(f"🛑 Daily SL atteint ({day_pnl:.2f}$). Pause 1h.")
            close_positions_by_magic()
            time.sleep(3600)
            continue

        # ── 2. Trailing stops ─────────────────────────────────
        update_trailing_stops()

        # ── 3. Biais H1 (recalculé une fois par heure) ────────
        # CORRECTION : vrai H1 indépendant du M15
        if last_h1_update is None or (now - last_h1_update).seconds >= 3600:
            current_h1_bias = get_trend(TIMEFRAME_H1, WINDOW_SMA_H1)
            last_h1_update  = now

        # ── 4. Biais M15 (recalculé à chaque cycle) ──────────
        current_m15_bias = get_trend(TIMEFRAME_M15, WINDOW_SMA_M15)

        # ── 5. Prédiction M15 (toutes les 15 min) ────────────
        if now.minute % 15 == 0 and 2 <= now.second < 7:
            df_live, X_m5, X_m15, X_h1, X_h4, X_reg = prepare_live_features_multiTF(
                checkpoint, WINDOW_SIZE_M15,
                "15min","30min","1h","4h","m15","m30","h1","h4")
            if df_live is not None:
                with torch.no_grad():
                    probs = F.softmax(model1(X_m5,X_m15,X_h1,X_h4,X_reg), dim=1).numpy()[0]
                confiance_buy_m15, confiance_sell_m15 = probs[0], probs[1]
            else:
                print("⚠ M15 : données insuffisantes")

        # ── 6. Prédiction M5 + décision check_confirm ────────
        if now.minute % 5 == 0 and 2 <= now.second < 7:
            df_live, X_m5, X_m15, X_h1, X_h4, X_reg = prepare_live_features_multiTF(
                checkpoint2, WINDOW_SIZE_M5,
                "5min","15min","1h","4h","m5","m15","h1","h4")
            if df_live is None:
                print("⚠️ M5 : données insuffisantes")
                time.sleep(5); continue

            with torch.no_grad():
                probs = F.softmax(model2(X_m5,X_m15,X_h1,X_h4,X_reg), dim=1).numpy()[0]
            confiance_buy_m5, confiance_sell_m5 = probs[1], probs[2]

            positions = get_current_positions()
            num_pos   = len(positions)

            # Réinitialiser check_confirm à chaque cycle M5
            check_confirm      = False
            check_confirm_time = None

            # Les deux filtres de tendance doivent être alignés
            # H1 fixe la direction macro, M15 confirme localement
            direction_ok = (current_h1_bias == current_m15_bias) and (current_m15_bias != 0)

            if direction_ok:
                if current_m15_bias == 1:
                    print(f"[{now.strftime('%H:%M')}] 📈 Haussier H1+M15 | "
                          f"M15={confiance_buy_m15:.3f}≥{LONG_T_m15} | "
                          f"M5={confiance_buy_m5:.3f}≥{LONG_T_m5} | PnL={day_pnl:.1f}$ | pos={num_pos}")
                    if confiance_buy_m5 >= LONG_T_m5 and confiance_buy_m15 >= LONG_T_m15:
                        check_confirm      = True
                        check_confirm_time = now
                else:
                    print(f"[{now.strftime('%H:%M')}] 📉 Baissier H1+M15 | "
                          f"M15={confiance_sell_m15:.3f}≥{SHORT_T_m15} | "
                          f"M5={confiance_sell_m5:.3f}≥{SHORT_T_m5} | PnL={day_pnl:.1f}$ | pos={num_pos}")
                    if confiance_sell_m5 >= SHORT_T_m5 and confiance_sell_m15 >= SHORT_T_m15:
                        check_confirm      = True
                        check_confirm_time = now
            else:
                print(f"[{now.strftime('%H:%M')}] ⏸  H1={current_h1_bias} M15={current_m15_bias} — pas d'alignement")

        # ── 7. Prédiction M1 + entrée (chaque minute si check_confirm) ──
        if 2 <= now.second < 7 and check_confirm:

            # NOUVEAU : expiration du check_confirm après 4 minutes
            # (évite d'entrer sur une bougie trop éloignée du signal M5)
            if check_confirm_time and (now - check_confirm_time).seconds > 240:
                check_confirm = False
                print("⌛ check_confirm expiré")
                time.sleep(5); continue

            df_live, X_m5, X_m15, X_h1, X_h4, X_reg = prepare_live_features_multiTF(
                checkpoint3, WINDOW_SIZE_M1,
                "1min","5min","15min","30min","m1","m5","m15","m30")
            if df_live is None:
                print("⚠️  M1 : données insuffisantes")
                time.sleep(5); continue

            with torch.no_grad():
                probs = F.softmax(model3(X_m5,X_m15,X_h1,X_h4,X_reg), dim=1).numpy()[0]
            confiance_buy_m1, confiance_sell_m1 = probs[1], probs[2]

            signal   = 0
            confiance = 0.0

            # BUY : H1 haussier + M1 confiant + filtre vol
            if (current_m15_bias == 1
                    and confiance_buy_m1 >= LONG_T_m1
                    and vol_filter_ok(df_live)):
                signal    = 1
                confiance = confiance_buy_m1
                print(f"  ➡️  M1 BUY signal | prob={confiance:.4f}")

            # SELL : H1 baissier + M1 confiant + filtre vol
            elif (current_m15_bias == -1
                    and confiance_sell_m1 >= SHORT_T_m1
                    and vol_filter_ok(df_live)):
                signal    = -1
                confiance = confiance_sell_m1
                print(f"  ➡️  M1 SELL signal | prob={confiance:.4f}")

            if signal == 0:
                time.sleep(5); continue

            if len(get_current_positions()) >= MAX_POSITIONS:
                print("⚠️  Max positions atteint.")
                time.sleep(5); continue

            # Calcul SL/TP et exécution
            sl_pts    = compute_sl_points_from_atr(df_live)
            rr        = compute_rr(df_live, confiance, current_m15_bias)
            direction = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
            print(f"  🎯 SL={sl_pts}pts | RR={rr:.2f}")
            execute_trade(direction, sl_pts, rr)

            # Réinitialiser après entrée
            check_confirm = False

        # ── 8. Pause ─────────────────────────────────────────
        time.sleep(5)

except KeyboardInterrupt:
    print("🛑 Arrêt manuel.")
    mt5.shutdown()
