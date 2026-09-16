"""
Relative Rotation Graph (RRG) - minimal free build
Needs: pip install yfinance pandas numpy matplotlib

Plots RS-Ratio (x) vs RS-Momentum (y) for a set of sectors/ETFs
against a benchmark, with trailing tails, in four quadrants:
Improving (top-left) -> Leading (top-right)
Lagging   (bottom-left) -> Weakening (bottom-right)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. CONFIGURE YOUR UNIVERSE ----
# Using NSE-listed sector ETFs (.NS) instead of raw index tickers -
# these have much more reliable data on Yahoo Finance than ^CNXxxx index symbols.
BENCHMARK = "NIFTYBEES.NS"   # Nifty 50 ETF
SECTORS = {
    "Bank":     "BANKBEES.NS",
    "IT":       "ITBEES.NS",
    "Auto":     "AUTOBEES.NS",
    "Pharma":   "PHARMABEES.NS",
    "FMCG":     "CONSUMBEES.NS",
    "PSU Bank": "PSUBNKBEES.NS",
    "Metal":    "METALIETF.NS",
    "Midcap":   "MID150BEES.NS",
}
# If any ETF above doesn't exist/trade, drop it from the dict -
# NSE ETF lineups change; check the symbol on nseindia.com or Yahoo Finance first.
LOOKBACK_PERIOD = "2y"        # history to download (weekly bars)
TAIL_LEN = 8                  # how many past weeks to trail per sector
RS_WINDOW = 14                 # window (in weeks) for z-score normalization
MOM_WINDOW = 10                 # weeks over which momentum (ROC) is measured
EMA_SPAN = 3                   # smoothing

# ---- 2. DOWNLOAD DATA ----
# Download each ticker separately, with retries and a short delay between
# requests - Yahoo Finance sometimes rate-limits requests from cloud/CI IPs
# (like GitHub Actions), so we retry a couple of times before giving up.
import time

tickers = list(SECTORS.values()) + [BENCHMARK]
series = {}
for t in tickers:
    ok = False
    for attempt in range(3):
        try:
            df = yf.Ticker(t).history(period=LOOKBACK_PERIOD, interval="1wk")
            if df is None or df.empty or "Close" not in df.columns:
                raise ValueError(f"no usable data (attempt {attempt+1})")
            s = df["Close"]
            if not isinstance(s, pd.Series) or s.dropna().empty:
                raise ValueError(f"unexpected/empty Close data (attempt {attempt+1})")
            series[t] = s
            ok = True
            break
        except Exception as e:
            print(f"attempt {attempt+1} for {t} failed: {e}")
            time.sleep(2)
    if not ok:
        print(f"WARNING: giving up on {t} after 3 attempts - skipping.")
    time.sleep(1)  # be gentle between tickers

print(f"Successfully downloaded: {list(series.keys())}")

if BENCHMARK not in series:
    raise SystemExit(f"Benchmark {BENCHMARK} failed to download - cannot continue.")

# Drop any sector whose ticker failed to download
SECTORS = {name: t for name, t in SECTORS.items() if t in series}
if not SECTORS:
    raise SystemExit("No sector tickers downloaded successfully - cannot continue.")

data = pd.concat(series, axis=1).dropna(how="all").ffill()
data.columns = list(series.keys())

# ---- 3. COMPUTE RS-RATIO AND RS-MOMENTUM (JdK-style) ----
def rs_ratio_momentum(sector_close, bench_close):
    rs = 100 * sector_close / bench_close
    rs_z = (rs - rs.rolling(RS_WINDOW).mean()) / rs.rolling(RS_WINDOW).std()
    rs_ratio = (100 + rs_z).ewm(span=EMA_SPAN).mean()

    roc = rs.pct_change(MOM_WINDOW) * 100
    mom_z = (roc - roc.rolling(RS_WINDOW).mean()) / roc.rolling(RS_WINDOW).std()
    rs_momentum = (100 + mom_z).ewm(span=EMA_SPAN).mean()

    return rs_ratio, rs_momentum

results = {}
for name, ticker in SECTORS.items():
    ratio, mom = rs_ratio_momentum(data[ticker], data[BENCHMARK])
    df = pd.DataFrame({"ratio": ratio, "momentum": mom}).dropna()
    if df.empty:
        print(f"WARNING: not enough data to plot {name} ({ticker}), skipping.")
        continue
    results[name] = df

if not results:
    raise SystemExit("No sectors had enough data to plot.")

# ---- 4. PLOT ----
fig, ax = plt.subplots(figsize=(9, 9))
ax.axhline(100, color="gray", lw=0.8)
ax.axvline(100, color="gray", lw=0.8)

quad_kwargs = dict(alpha=0.08)
ax.axhspan(100, 200, xmin=0.5, xmax=1, color="green", **quad_kwargs)   # Leading
ax.axhspan(100, 200, xmin=0, xmax=0.5, color="blue", **quad_kwargs)    # Improving
ax.axhspan(0, 100, xmin=0, xmax=0.5, color="red", **quad_kwargs)       # Lagging
ax.axhspan(0, 100, xmin=0.5, xmax=1, color="orange", **quad_kwargs)    # Weakening

colors = plt.cm.tab10.colors
for i, (name, df) in enumerate(results.items()):
    tail = df.tail(TAIL_LEN)
    c = colors[i % len(colors)]
    ax.plot(tail["ratio"], tail["momentum"], "-", color=c, lw=1.5, alpha=0.7)
    ax.scatter(tail["ratio"][:-1], tail["momentum"][:-1], color=c, s=20, alpha=0.5)
    ax.scatter(tail["ratio"].iloc[-1], tail["momentum"].iloc[-1], color=c, s=90,
               edgecolor="black", zorder=5, label=name)
    ax.annotate(name, (tail["ratio"].iloc[-1], tail["momentum"].iloc[-1]),
                fontsize=9, fontweight="bold", xytext=(5, 5), textcoords="offset points")

ax.set_xlabel("RS-Ratio")
ax.set_ylabel("RS-Momentum")
ax.set_title("Sector Rotation - Relative Rotation Graph")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("rrg_chart.png", dpi=150)
plt.show()
