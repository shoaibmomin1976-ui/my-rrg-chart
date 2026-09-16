"""
Interactive Relative Rotation Graph (RRG) - free build, with Daily/Weekly toggle
Needs: pip install yfinance pandas numpy plotly

Outputs docs/index.html - an interactive, zoomable, hoverable RRG chart
with a Daily/Weekly toggle, servable for free via GitHub Pages
(Settings -> Pages -> Deploy from branch "main", folder "/docs").
"""

import time
import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---- 1. CONFIGURE YOUR UNIVERSE ----
BENCHMARK = "NIFTYBEES.NS"   # Nifty 50 ETF
SECTORS = {
    "Bank":      "BANKBEES.NS",
    "IT":        "ITBEES.NS",
    "Auto":      "AUTOBEES.NS",
    "Pharma":    "PHARMABEES.NS",
    "FMCG":      "CONSUMBEES.NS",
    "PSU Bank":  "PSUBNKBEES.NS",
    "Metal":     "METALIETF.NS",
    "Midcap":    "MID150BEES.NS",
    "Infra":     "INFRABEES.NS",
    "CPSE":      "CPSEETF.NS",
    "Next 50":   "JUNIORBEES.NS",
}

# Separate settings per timeframe
TIMEFRAMES = {
    "Weekly": dict(period="2y", interval="1wk", rs_window=14, mom_window=10, tail_len=8, ema_span=3),
    "Daily":  dict(period="1y", interval="1d",  rs_window=20, mom_window=10, tail_len=20, ema_span=3),
}

COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
          "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8"]

# ---- 2. DOWNLOAD HELPER (with retries - Yahoo can rate-limit cloud IPs) ----
def download_close(ticker, period, interval):
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if df is None or df.empty or "Close" not in df.columns:
                raise ValueError(f"no usable data (attempt {attempt+1})")
            s = df["Close"]
            if not isinstance(s, pd.Series) or s.dropna().empty:
                raise ValueError(f"unexpected/empty Close data (attempt {attempt+1})")
            return s
        except Exception as e:
            print(f"attempt {attempt+1} for {ticker} ({interval}) failed: {e}")
            time.sleep(2)
    print(f"WARNING: giving up on {ticker} ({interval}) after 3 attempts - skipping.")
    return None

def rs_ratio_momentum(sector_close, bench_close, rs_window, mom_window, ema_span):
    rs = 100 * sector_close / bench_close
    rs_z = (rs - rs.rolling(rs_window).mean()) / rs.rolling(rs_window).std()
    rs_ratio = (100 + rs_z).ewm(span=ema_span).mean()

    roc = rs.pct_change(mom_window) * 100
    mom_z = (roc - roc.rolling(rs_window).mean()) / roc.rolling(rs_window).std()
    rs_momentum = (100 + mom_z).ewm(span=ema_span).mean()

    return rs_ratio, rs_momentum

# ---- 3. BUILD DATA + TRACES FOR EACH TIMEFRAME ----
fig = go.Figure()
traces_per_timeframe = {}   # tf_name -> list of trace indices in fig.data
all_x, all_y = [], []       # for shared axis range across both timeframes

for tf_name, cfg in TIMEFRAMES.items():
    tickers = list(SECTORS.values()) + [BENCHMARK]
    series = {}
    for t in tickers:
        s = download_close(t, cfg["period"], cfg["interval"])
        if s is not None:
            series[t] = s
        time.sleep(1)

    if BENCHMARK not in series:
        print(f"WARNING: benchmark missing for {tf_name}, skipping this timeframe.")
        traces_per_timeframe[tf_name] = []
        continue

    tf_sectors = {name: t for name, t in SECTORS.items() if t in series}
    if not tf_sectors:
        print(f"WARNING: no sectors available for {tf_name}, skipping.")
        traces_per_timeframe[tf_name] = []
        continue

    data = pd.concat(series, axis=1).dropna(how="all").ffill()
    data.columns = list(series.keys())

    results = {}
    for name, ticker in tf_sectors.items():
        ratio, mom = rs_ratio_momentum(data[ticker], data[BENCHMARK],
                                        cfg["rs_window"], cfg["mom_window"], cfg["ema_span"])
        df = pd.DataFrame({"ratio": ratio, "momentum": mom}).dropna()
        if df.empty:
            print(f"WARNING: not enough {tf_name} data for {name}, skipping.")
            continue
        results[name] = df

    trace_indices = []
    for i, (name, df) in enumerate(results.items()):
        tail = df.tail(cfg["tail_len"])
        c = COLORS[i % len(COLORS)]
        all_x.extend(tail["ratio"].tolist())
        all_y.extend(tail["momentum"].tolist())

        fig.add_trace(go.Scatter(
            x=tail["ratio"], y=tail["momentum"],
            mode="lines+markers",
            name=name,
            legendgroup=name,
            line=dict(color=c, width=2, shape="spline"),
            marker=dict(size=6, color=c),
            visible=(tf_name == "Weekly"),
            hovertemplate=f"<b>{name}</b> ({tf_name})<br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>",
        ))
        trace_indices.append(len(fig.data) - 1)

        fig.add_trace(go.Scatter(
            x=[tail["ratio"].iloc[-1]], y=[tail["momentum"].iloc[-1]],
            mode="markers+text",
            marker=dict(size=12, color=c, line=dict(color="black", width=1)),
            text=[name], textposition="top center",
            legendgroup=name,
            showlegend=False,
            visible=(tf_name == "Weekly"),
            hovertemplate=f"<b>{name}</b> (latest, {tf_name})<br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>",
        ))
        trace_indices.append(len(fig.data) - 1)

    traces_per_timeframe[tf_name] = trace_indices

if not any(traces_per_timeframe.values()):
    raise SystemExit("No data available for either timeframe - cannot build chart.")

# ---- 4. SHARED AXIS RANGE (covers both daily & weekly so toggle doesn't jump) ----
all_x = np.array(all_x) if all_x else np.array([99, 101])
all_y = np.array(all_y) if all_y else np.array([99, 101])
pad_x = max(0.3, (all_x.max() - all_x.min()) * 0.2)
pad_y = max(0.3, (all_y.max() - all_y.min()) * 0.2)
x_min, x_max = all_x.min() - pad_x, all_x.max() + pad_x
y_min, y_max = all_y.min() - pad_y, all_y.max() + pad_y

# Quadrant backgrounds
fig.add_shape(type="rect", x0=100, x1=x_max, y0=100, y1=y_max,
              fillcolor="green", opacity=0.08, line_width=0, layer="below")
fig.add_shape(type="rect", x0=x_min, x1=100, y0=100, y1=y_max,
              fillcolor="blue", opacity=0.08, line_width=0, layer="below")
fig.add_shape(type="rect", x0=x_min, x1=100, y0=y_min, y1=100,
              fillcolor="red", opacity=0.08, line_width=0, layer="below")
fig.add_shape(type="rect", x0=100, x1=x_max, y0=y_min, y1=100,
              fillcolor="orange", opacity=0.08, line_width=0, layer="below")

fig.add_annotation(x=x_min + pad_x*0.3, y=y_max - pad_y*0.15, text="Improving",
                    showarrow=False, font=dict(color="blue", size=13))
fig.add_annotation(x=x_max - pad_x*0.5, y=y_max - pad_y*0.15, text="Leading",
                    showarrow=False, font=dict(color="green", size=13))
fig.add_annotation(x=x_min + pad_x*0.3, y=y_min + pad_y*0.15, text="Lagging",
                    showarrow=False, font=dict(color="red", size=13))
fig.add_annotation(x=x_max - pad_x*0.5, y=y_min + pad_y*0.15, text="Weakening",
                    showarrow=False, font=dict(color="orange", size=13))

fig.add_hline(y=100, line_color="gray", line_width=1)
fig.add_vline(x=100, line_color="gray", line_width=1)

# ---- 5. DAILY / WEEKLY TOGGLE BUTTONS ----
n_traces = len(fig.data)
buttons = []
for tf_name, idxs in traces_per_timeframe.items():
    visibility = [False] * n_traces
    for idx in idxs:
        visibility[idx] = True
    buttons.append(dict(
        label=tf_name,
        method="update",
        args=[{"visible": visibility},
              {"title": f"Sector Rotation - Relative Rotation Graph ({tf_name})"}],
    ))

fig.update_layout(
    title="Sector Rotation - Relative Rotation Graph (Weekly)",
    xaxis_title="RS-Ratio",
    yaxis_title="RS-Momentum",
    xaxis=dict(range=[x_min, x_max]),
    yaxis=dict(range=[y_min, y_max]),
    template="plotly_white",
    width=1000, height=800,
    legend=dict(orientation="v", x=1.02, y=1),
    updatemenus=[dict(
        type="buttons",
        direction="right",
        active=0,
        x=0.0, y=1.12,
        xanchor="left", yanchor="top",
        buttons=buttons,
    )],
)

os.makedirs("docs", exist_ok=True)
fig.write_html("docs/index.html", include_plotlyjs="cdn")
print("Saved interactive chart with Daily/Weekly toggle to docs/index.html")
