"""
Interactive Relative Rotation Graph (RRG) - free build
Needs: pip install yfinance pandas numpy plotly

Outputs docs/index.html - servable free via GitHub Pages
(Settings -> Pages -> Deploy from branch "main", folder "/docs").

Two controls on the chart:
- Timeframe buttons (Weekly / Daily) - apply to the "All Sectors" overview.
- "View" dropdown - switch between the sector overview and a drill-down
  into individual stocks within one sector (vs that sector's own index).
"""

import time
import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---- 1. CONFIGURE UNIVERSE ----
BENCHMARK = "NIFTYBEES.NS"   # Nifty 50 ETF - overall market benchmark

# Sector overview universe. Some entries use best-effort NSE index tickers
# that may not always resolve on Yahoo - the download step below skips any
# that fail instead of crashing the whole chart.
SECTORS = {
    "Bank":          "BANKBEES.NS",
    "IT":            "ITBEES.NS",
    "Auto":          "AUTOBEES.NS",
    "Pharma":        "PHARMABEES.NS",
    "FMCG":          "CONSUMBEES.NS",
    "PSU Bank":      "PSUBNKBEES.NS",
    "Metal":         "METALIETF.NS",
    "Midcap":        "MID150BEES.NS",
    "Infra":         "INFRABEES.NS",
    "CPSE":          "CPSEETF.NS",
    "Next 50":       "JUNIORBEES.NS",
    "Bank Nifty":    "^NSEBANK",
    "Private Bank":  "NIFTYPVTBANK.NS",
    "Fin Services":  "^CNXFIN",
    "Sensex":        "^BSESN",
    "Realty":        "^CNXREALTY",
    "Media":         "^CNXMEDIA",
    "Energy":        "^CNXENERGY",
    "Healthcare":    "^CNXHEALTH",
}

# Stock drill-down: for each sector, its own index/ETF is the benchmark for
# ranking its constituent stocks against each other.
STOCK_SECTORS = {
    "Bank": ("BANKBEES.NS", {
        "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "SBI": "SBIN.NS",
        "Axis Bank": "AXISBANK.NS", "Kotak Bank": "KOTAKBANK.NS", "IndusInd": "INDUSINDBK.NS",
    }),
    "IT": ("ITBEES.NS", {
        "TCS": "TCS.NS", "Infosys": "INFY.NS", "Wipro": "WIPRO.NS",
        "HCL Tech": "HCLTECH.NS", "Tech Mahindra": "TECHM.NS",
    }),
    "Auto": ("AUTOBEES.NS", {
        "Maruti": "MARUTI.NS", "Tata Motors": "TATAMOTORS.NS", "M&M": "M&M.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS", "Hero MotoCorp": "HEROMOTOCO.NS",
    }),
    "Pharma": ("PHARMABEES.NS", {
        "Sun Pharma": "SUNPHARMA.NS", "Dr Reddy's": "DRREDDY.NS", "Cipla": "CIPLA.NS",
        "Divi's Lab": "DIVISLAB.NS", "Lupin": "LUPIN.NS",
    }),
    "FMCG": ("CONSUMBEES.NS", {
        "HUL": "HINDUNILVR.NS", "ITC": "ITC.NS", "Nestle India": "NESTLEIND.NS",
        "Britannia": "BRITANNIA.NS", "Dabur": "DABUR.NS",
    }),
    "Metal": ("METALIETF.NS", {
        "Tata Steel": "TATASTEEL.NS", "JSW Steel": "JSWSTEEL.NS", "Hindalco": "HINDALCO.NS",
        "Vedanta": "VEDL.NS", "SAIL": "SAIL.NS",
    }),
    "PSU Bank": ("PSUBNKBEES.NS", {
        "SBI": "SBIN.NS", "Bank of Baroda": "BANKBARODA.NS", "PNB": "PNB.NS",
        "Canara Bank": "CANBK.NS", "Union Bank": "UNIONBANK.NS",
    }),
}

TIMEFRAMES = {
    "Weekly": dict(period="2y", interval="1wk", rs_window=14, mom_window=10, tail_len=8, ema_span=3),
    "Daily":  dict(period="1y", interval="1d",  rs_window=20, mom_window=10, tail_len=20, ema_span=3),
}
STOCK_TF = TIMEFRAMES["Weekly"]  # stock drill-down views use weekly only, to keep scope manageable

COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
          "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8",
          "#393b79","#637939","#8c6d31","#843c39","#7b4173","#a55194","#ce6dbd"]

# ---- 2. DOWNLOAD HELPER (with retries - Yahoo can rate-limit cloud IPs) ----
_cache = {}
def download_close(ticker, period, interval):
    key = (ticker, period, interval)
    if key in _cache:
        return _cache[key]
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if df is None or df.empty or "Close" not in df.columns:
                raise ValueError(f"no usable data (attempt {attempt+1})")
            s = df["Close"]
            if not isinstance(s, pd.Series) or s.dropna().empty:
                raise ValueError(f"unexpected/empty Close data (attempt {attempt+1})")
            _cache[key] = s
            return s
        except Exception as e:
            print(f"attempt {attempt+1} for {ticker} ({interval}) failed: {e}")
            time.sleep(2)
    print(f"WARNING: giving up on {ticker} ({interval}) after 3 attempts - skipping.")
    _cache[key] = None
    return None

def rs_ratio_momentum(sector_close, bench_close, rs_window, mom_window, ema_span):
    rs = 100 * sector_close / bench_close
    rs_z = (rs - rs.rolling(rs_window).mean()) / rs.rolling(rs_window).std()
    rs_ratio = (100 + rs_z).ewm(span=ema_span).mean()

    roc = rs.pct_change(mom_window) * 100
    mom_z = (roc - roc.rolling(rs_window).mean()) / roc.rolling(rs_window).std()
    rs_momentum = (100 + mom_z).ewm(span=ema_span).mean()

    return rs_ratio, rs_momentum

def build_results(items, benchmark_ticker, period, interval, rs_window, mom_window, tail_len, ema_span):
    """items: dict name -> ticker. Returns dict name -> tail dataframe(ratio, momentum)."""
    bench_s = download_close(benchmark_ticker, period, interval)
    if bench_s is None:
        return {}
    out = {}
    for name, ticker in items.items():
        s = download_close(ticker, period, interval)
        time.sleep(0.5)
        if s is None:
            continue
        combined = pd.concat({"s": s, "b": bench_s}, axis=1).dropna(how="all").ffill().dropna()
        if combined.empty or len(combined) < rs_window + mom_window:
            print(f"WARNING: not enough overlapping data for {name}, skipping.")
            continue
        ratio, mom = rs_ratio_momentum(combined["s"], combined["b"], rs_window, mom_window, ema_span)
        df = pd.DataFrame({"ratio": ratio, "momentum": mom}).dropna()
        if df.empty:
            continue
        out[name] = df.tail(tail_len)
    return out

# ---- 3. BUILD ALL TRACE GROUPS ----
fig = go.Figure()
groups = {}          # group_key -> list of trace indices
all_x, all_y = [], []

def add_group(group_key, results_dict, visible_default=False):
    idxs = []
    for i, (name, tail) in enumerate(results_dict.items()):
        c = COLORS[i % len(COLORS)]
        all_x.extend(tail["ratio"].tolist())
        all_y.extend(tail["momentum"].tolist())
        fig.add_trace(go.Scatter(
            x=tail["ratio"], y=tail["momentum"], mode="lines+markers", name=name,
            legendgroup=f"{group_key}-{name}", line=dict(color=c, width=2, shape="spline"),
            marker=dict(size=6, color=c), visible=visible_default,
            hovertemplate=f"<b>{name}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>",
        ))
        idxs.append(len(fig.data) - 1)
        fig.add_trace(go.Scatter(
            x=[tail["ratio"].iloc[-1]], y=[tail["momentum"].iloc[-1]], mode="markers+text",
            marker=dict(size=12, color=c, line=dict(color="black", width=1)),
            text=[name], textposition="top center", legendgroup=f"{group_key}-{name}",
            showlegend=False, visible=visible_default,
            hovertemplate=f"<b>{name}</b> (latest)<br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>",
        ))
        idxs.append(len(fig.data) - 1)
    groups[group_key] = idxs

# Sector overview - Weekly (default view)
weekly_cfg = TIMEFRAMES["Weekly"]
sector_weekly = build_results(SECTORS, BENCHMARK, weekly_cfg["period"], weekly_cfg["interval"],
                               weekly_cfg["rs_window"], weekly_cfg["mom_window"],
                               weekly_cfg["tail_len"], weekly_cfg["ema_span"])
add_group("sector_weekly", sector_weekly, visible_default=True)

# Sector overview - Daily
daily_cfg = TIMEFRAMES["Daily"]
sector_daily = build_results(SECTORS, BENCHMARK, daily_cfg["period"], daily_cfg["interval"],
                              daily_cfg["rs_window"], daily_cfg["mom_window"],
                              daily_cfg["tail_len"], daily_cfg["ema_span"])
add_group("sector_daily", sector_daily, visible_default=False)

# Stock drill-downs (weekly only)
for sector_name, (bench_ticker, stocks) in STOCK_SECTORS.items():
    res = build_results(stocks, bench_ticker, STOCK_TF["period"], STOCK_TF["interval"],
                         STOCK_TF["rs_window"], STOCK_TF["mom_window"],
                         STOCK_TF["tail_len"], STOCK_TF["ema_span"])
    add_group(f"stocks_{sector_name}", res, visible_default=False)

if not groups.get("sector_weekly") and not groups.get("sector_daily"):
    raise SystemExit("No sector data available at all - cannot build chart.")

# ---- 4. SHARED AXIS RANGE ----
all_x = np.array(all_x) if all_x else np.array([99, 101])
all_y = np.array(all_y) if all_y else np.array([99, 101])
pad_x = max(0.3, (all_x.max() - all_x.min()) * 0.2)
pad_y = max(0.3, (all_y.max() - all_y.min()) * 0.2)
x_min, x_max = all_x.min() - pad_x, all_x.max() + pad_x
y_min, y_max = all_y.min() - pad_y, all_y.max() + pad_y

fig.add_shape(type="rect", x0=100, x1=x_max, y0=100, y1=y_max, fillcolor="green", opacity=0.08, line_width=0, layer="below")
fig.add_shape(type="rect", x0=x_min, x1=100, y0=100, y1=y_max, fillcolor="blue", opacity=0.08, line_width=0, layer="below")
fig.add_shape(type="rect", x0=x_min, x1=100, y0=y_min, y1=100, fillcolor="red", opacity=0.08, line_width=0, layer="below")
fig.add_shape(type="rect", x0=100, x1=x_max, y0=y_min, y1=100, fillcolor="orange", opacity=0.08, line_width=0, layer="below")

fig.add_annotation(x=x_min + pad_x*0.3, y=y_max - pad_y*0.15, text="Improving", showarrow=False, font=dict(color="blue", size=13))
fig.add_annotation(x=x_max - pad_x*0.5, y=y_max - pad_y*0.15, text="Leading", showarrow=False, font=dict(color="green", size=13))
fig.add_annotation(x=x_min + pad_x*0.3, y=y_min + pad_y*0.15, text="Lagging", showarrow=False, font=dict(color="red", size=13))
fig.add_annotation(x=x_max - pad_x*0.5, y=y_min + pad_y*0.15, text="Weakening", showarrow=False, font=dict(color="orange", size=13))

fig.add_hline(y=100, line_color="gray", line_width=1)
fig.add_vline(x=100, line_color="gray", line_width=1)

# ---- 5. CONTROLS ----
n_traces = len(fig.data)

def visibility_for(active_group_key):
    v = [False] * n_traces
    for idx in groups.get(active_group_key, []):
        v[idx] = True
    return v

# Timeframe buttons (only meaningful for sector overview)
timeframe_buttons = [
    dict(label="Weekly", method="update",
         args=[{"visible": visibility_for("sector_weekly")},
               {"title": "Sector Rotation - All Sectors (Weekly)"}]),
    dict(label="Daily", method="update",
         args=[{"visible": visibility_for("sector_daily")},
               {"title": "Sector Rotation - All Sectors (Daily)"}]),
]

# View dropdown: All Sectors + one entry per stock drill-down
view_buttons = [
    dict(label="All Sectors", method="update",
         args=[{"visible": visibility_for("sector_weekly")},
               {"title": "Sector Rotation - All Sectors (Weekly)"}]),
]
for sector_name in STOCK_SECTORS.keys():
    key = f"stocks_{sector_name}"
    if groups.get(key):
        view_buttons.append(dict(
            label=f"{sector_name} Stocks", method="update",
            args=[{"visible": visibility_for(key)},
                  {"title": f"{sector_name} Sector - Stock Rotation (Weekly)"}],
        ))

fig.update_layout(
    title="Sector Rotation - All Sectors (Weekly)",
    xaxis_title="RS-Ratio",
    yaxis_title="RS-Momentum",
    xaxis=dict(range=[x_min, x_max]),
    yaxis=dict(range=[y_min, y_max]),
    template="plotly_white",
    width=1050, height=800,
    legend=dict(orientation="v", x=1.02, y=1),
    updatemenus=[
        dict(type="buttons", direction="right", active=0, x=0.0, y=1.12,
             xanchor="left", yanchor="top", buttons=timeframe_buttons),
        dict(type="dropdown", direction="down", active=0, x=0.25, y=1.12,
             xanchor="left", yanchor="top", buttons=view_buttons),
    ],
)

os.makedirs("docs", exist_ok=True)
fig.write_html("docs/index.html", include_plotlyjs="cdn")
print("Saved chart to docs/index.html")
print(f"Groups built: {[(k, len(v)) for k, v in groups.items()]}")
