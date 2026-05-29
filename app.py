
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="SSI Trading Analytics Dashboard", layout="wide")

# =========================================================
# Formatting helpers
# =========================================================

def fmt_vnd(x):
    if pd.isna(x):
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    sign = "-" if x < 0 else ""
    x = abs(round(x))
    return sign + f"{x:,.0f}".replace(",", ".")

def fmt_pct(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)

def fmt_num(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):,.0f}".replace(",", ".")
    except Exception:
        return str(x)

def style_money_cols(df):
    out = df.copy()
    for c in out.columns:
        cl = c.lower()
        if any(k in cl for k in ["pnl", "value", "gain", "loss", "missed", "cash", "fee", "tax", "price", "total"]):
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(fmt_vnd)
        elif "rate" in cl or "return" in cl or "increase_pct" in cl or "decrease_pct" in cl:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(fmt_pct)
        elif any(k in cl for k in ["quantity", "days", "count", "trades"]):
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(fmt_num)
    return out

# =========================================================
# Input normalization
# =========================================================

def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)

def normalize_order_history(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Expected preferred format:
    # date, datetime, ticker, side, quantity, price, fee, tax, gross_value, net_cash_flow
    required = {"date", "ticker", "side", "quantity", "price"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"File thiếu cột bắt buộc: {missing}")
        st.write("Các cột tìm thấy:", list(df.columns))
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df["datetime"] = df["date"]

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["side"] = df["side"].astype(str).str.upper().str.strip()

    # Accept MUA/BAN if side_vi somehow got mapped
    df["side"] = df["side"].replace({
        "MUA": "BUY", "BÁN": "SELL", "BAN": "SELL",
        "BUY": "BUY", "SELL": "SELL",
        "B": "BUY", "S": "SELL"
    })

    for col in ["quantity", "price", "fee", "tax", "gross_value", "net_cash_flow", "total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    df = df.dropna(subset=["date", "ticker", "side", "quantity", "price"])
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)
    return df

# =========================================================
# FIFO realized trade matching
# =========================================================

def fifo_match_trades(trades):
    lots = {}
    realized_rows = []

    for _, r in trades.iterrows():
        ticker = r["ticker"]
        side = r["side"]
        qty = float(r["quantity"])
        price = float(r["price"])
        fee = float(r.get("fee", 0) or 0)
        tax = float(r.get("tax", 0) or 0)
        dt = r["date"]
        dttm = r.get("datetime", dt)

        lots.setdefault(ticker, [])

        if side == "BUY":
            lots[ticker].append({
                "buy_date": dt,
                "buy_datetime": dttm,
                "buy_price": price,
                "remaining_qty": qty,
                "buy_fee_per_share": fee / qty if qty else 0,
            })

        elif side == "SELL":
            remaining_sell = qty
            sell_fee_per_share = fee / qty if qty else 0
            sell_tax_per_share = tax / qty if qty else 0

            while remaining_sell > 0 and lots[ticker]:
                lot = lots[ticker][0]
                matched_qty = min(remaining_sell, lot["remaining_qty"])

                cost_per_share = lot["buy_price"] + lot.get("buy_fee_per_share", 0)
                proceeds_per_share = price - sell_fee_per_share - sell_tax_per_share

                actual_pnl = (proceeds_per_share - cost_per_share) * matched_qty
                actual_return = (proceeds_per_share / cost_per_share - 1) if cost_per_share else np.nan
                holding_days = (dt - lot["buy_date"]).days

                realized_rows.append({
                    "ticker": ticker,
                    "buy_date": lot["buy_date"],
                    "sell_date": dt,
                    "sell_datetime": dttm,
                    "quantity": matched_qty,
                    "buy_price": lot["buy_price"],
                    "sell_price": price,
                    "cost_value": cost_per_share * matched_qty,
                    "sell_value": proceeds_per_share * matched_qty,
                    "actual_pnl": actual_pnl,
                    "actual_return": actual_return,
                    "holding_days": holding_days,
                    "sell_fee": sell_fee_per_share * matched_qty,
                    "sell_tax": sell_tax_per_share * matched_qty,
                })

                lot["remaining_qty"] -= matched_qty
                remaining_sell -= matched_qty

                if lot["remaining_qty"] <= 0.000001:
                    lots[ticker].pop(0)

            # If there is a sell without enough prior buys, keep an unmatched row
            if remaining_sell > 0:
                realized_rows.append({
                    "ticker": ticker,
                    "buy_date": pd.NaT,
                    "sell_date": dt,
                    "sell_datetime": dttm,
                    "quantity": remaining_sell,
                    "buy_price": np.nan,
                    "sell_price": price,
                    "cost_value": np.nan,
                    "sell_value": price * remaining_sell,
                    "actual_pnl": np.nan,
                    "actual_return": np.nan,
                    "holding_days": np.nan,
                    "sell_fee": sell_fee_per_share * remaining_sell,
                    "sell_tax": sell_tax_per_share * remaining_sell,
                })

    return pd.DataFrame(realized_rows)

# =========================================================
# Price data
# =========================================================

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_yahoo_prices(tickers, start_date, end_date):
    if yf is None:
        return pd.DataFrame()

    frames = []
    for ticker in tickers:
        yahoo_ticker = f"{ticker}.VN"
        try:
            hist = yf.download(
                yahoo_ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if hist is None or hist.empty:
                continue
            hist = hist.reset_index()
            close_col = "Close"
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]
            out = hist[["Date", close_col]].rename(columns={"Date": "date", close_col: "close"})
            out["ticker"] = ticker
            out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
            out["close"] = pd.to_numeric(out["close"], errors="coerce")
            frames.append(out[["date", "ticker", "close"]].dropna())
        except Exception:
            continue

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def normalize_price_history(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"date", "ticker", "close"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"File giá thiếu cột: {missing}. Cần có: date,ticker,close")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])

def enrich_with_exit_analysis(realized, prices, evaluation_days):
    if realized.empty:
        return realized

    rows = []
    for _, r in realized.iterrows():
        row = r.to_dict()
        ticker = r["ticker"]
        sell_date = r["sell_date"]
        buy_date = r["buy_date"]
        sell_price = r["sell_price"]
        buy_price = r["buy_price"]
        qty = r["quantity"]

        ticker_prices = prices[prices["ticker"] == ticker].copy()

        row.update({
            "peak_price": np.nan,
            "peak_date": pd.NaT,
            "potential_pnl": np.nan,
            "potential_return": np.nan,
            "potential_minus_actual": np.nan,
            "capture_rate": np.nan,
            "days_to_peak_after_buy": np.nan,
            "days_to_peak_after_sell": np.nan,
            "recommended_holding_days": np.nan,
            "max_price_after_sell": np.nan,
            "min_price_after_sell": np.nan,
            "increase_after_sell_pct": np.nan,
            "decrease_after_sell_pct": np.nan,
        })

        if ticker_prices.empty or pd.isna(sell_date):
            rows.append(row)
            continue

        # Window for exit quality: from buy date to sell date + evaluation days.
        # If buy date is missing, use sell date as start.
        window_start = buy_date if pd.notna(buy_date) else sell_date
        window_end = sell_date + timedelta(days=evaluation_days)

        full_window = ticker_prices[
            (ticker_prices["date"] >= window_start) &
            (ticker_prices["date"] <= window_end)
        ].sort_values("date")

        after_sell = ticker_prices[
            (ticker_prices["date"] > sell_date) &
            (ticker_prices["date"] <= window_end)
        ].sort_values("date")

        if not full_window.empty:
            peak_idx = full_window["close"].idxmax()
            peak = full_window.loc[peak_idx]
            peak_price = float(peak["close"])
            peak_date = peak["date"]

            row["peak_price"] = peak_price
            row["peak_date"] = peak_date

            if pd.notna(buy_price):
                potential_pnl = (peak_price - buy_price) * qty
                row["potential_pnl"] = potential_pnl
                row["potential_return"] = peak_price / buy_price - 1 if buy_price else np.nan
                row["potential_minus_actual"] = potential_pnl - r["actual_pnl"] if pd.notna(r["actual_pnl"]) else np.nan

                max_possible_profit = max(potential_pnl, 0)
                actual_profit = max(r["actual_pnl"], 0) if pd.notna(r["actual_pnl"]) else np.nan
                row["capture_rate"] = actual_profit / max_possible_profit if max_possible_profit > 0 else np.nan

                row["days_to_peak_after_buy"] = (peak_date - buy_date).days if pd.notna(buy_date) else np.nan
                row["recommended_holding_days"] = row["days_to_peak_after_buy"]

            row["days_to_peak_after_sell"] = (peak_date - sell_date).days if pd.notna(sell_date) else np.nan

        if not after_sell.empty:
            max_after = float(after_sell["close"].max())
            min_after = float(after_sell["close"].min())
            row["max_price_after_sell"] = max_after
            row["min_price_after_sell"] = min_after
            row["increase_after_sell_pct"] = max_after / sell_price - 1 if sell_price else np.nan
            row["decrease_after_sell_pct"] = min_after / sell_price - 1 if sell_price else np.nan

        rows.append(row)

    return pd.DataFrame(rows)

# =========================================================
# Performance metrics
# =========================================================

def compute_metrics(df):
    if df.empty:
        return {}
    valid = df.dropna(subset=["actual_pnl"])
    wins = valid[valid["actual_pnl"] > 0]
    losses = valid[valid["actual_pnl"] < 0]

    win_rate = len(wins) / len(valid) if len(valid) else 0
    avg_win = wins["actual_return"].mean() if len(wins) else 0
    avg_loss = abs(losses["actual_return"].mean()) if len(losses) else 0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    gross_profit = wins["actual_pnl"].sum()
    gross_loss = abs(losses["actual_pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    equity = valid.sort_values("sell_date")["actual_pnl"].cumsum()
    rolling_max = equity.cummax()
    drawdown = equity - rolling_max
    max_drawdown = drawdown.min() if len(drawdown) else 0

    # Consecutive losses
    max_consec = 0
    cur = 0
    for pnl in valid.sort_values("sell_date")["actual_pnl"]:
        if pnl < 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    return {
        "total_actual_pnl": valid["actual_pnl"].sum(),
        "total_potential_pnl": valid["potential_pnl"].sum(skipna=True),
        "total_missed_pnl": valid["potential_minus_actual"].sum(skipna=True),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "consecutive_losses": max_consec,
        "avg_holding_days": valid["holding_days"].mean(),
        "avg_recommended_holding_days": valid["recommended_holding_days"].mean(skipna=True),
    }

# =========================================================
# UI
# =========================================================

st.title("SSI Trading Analytics Dashboard")
st.caption("Built for cleaned_order_history_dashboard_format.csv — stocks first, with room for futures/market-context analysis later.")

with st.sidebar:
    st.header("Input")
    trades_file = st.file_uploader("Upload order history CSV/XLSX", type=["csv", "xlsx", "xls"])
    price_mode = st.radio("Price source", ["Auto Yahoo Finance", "Upload price_history.csv"], index=0)
    price_file = None
    if price_mode == "Upload price_history.csv":
        price_file = st.file_uploader("Upload price_history.csv", type=["csv", "xlsx", "xls"])

    evaluation_days = st.selectbox(
        "Potential/peak window after sell",
        options=[7, 14, 30, 60, 90, 180],
        index=2,
        help="Dashboard looks for the best price from buy date until sell date + this many days."
    )

    st.header("Optional manual labels")
    st.caption("These are not required. Add these columns later if you want deeper analysis.")
    st.code("setup,market_condition,entry_trigger,stop_loss,target,behavior_tag", language="text")

if trades_file is None:
    st.info("Upload `cleaned_order_history_dashboard_format.csv` to start.")
    st.stop()

raw = read_uploaded_file(trades_file)
trades = normalize_order_history(raw)

st.success(f"Loaded {len(trades)} standardized order rows.")

realized = fifo_match_trades(trades)

if realized.empty:
    st.warning("No realized sell trades found. Need BUY and SELL rows to compute actual P&L.")
    st.stop()

min_date = trades["date"].min() - timedelta(days=10)
max_date = trades["date"].max() + timedelta(days=evaluation_days + 10)
tickers = sorted(trades["ticker"].dropna().unique())

prices = pd.DataFrame()
if price_mode == "Auto Yahoo Finance":
    with st.spinner("Fetching historical prices from Yahoo Finance..."):
        prices = fetch_yahoo_prices(tickers, min_date, max_date)
else:
    if price_file is None:
        st.warning("Upload price_history.csv with columns: date,ticker,close")
        st.stop()
    price_raw = read_uploaded_file(price_file)
    prices = normalize_price_history(price_raw)

if prices.empty:
    st.warning("No price history loaded. Actual P&L still works, but Potential P&L / peak analysis needs price history.")
else:
    st.success(f"Loaded price history for {prices['ticker'].nunique()} tickers.")

analysis = enrich_with_exit_analysis(realized, prices, evaluation_days)
metrics = compute_metrics(analysis)

# =========================================================
# Top metrics
# =========================================================

st.header("1. Executive summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Actual P&L", fmt_vnd(metrics.get("total_actual_pnl", 0)))
c2.metric("Potential P&L", fmt_vnd(metrics.get("total_potential_pnl", 0)))
c3.metric("Missed / Difference", fmt_vnd(metrics.get("total_missed_pnl", 0)))
c4.metric("Win rate", fmt_pct(metrics.get("win_rate", 0)))

c5, c6, c7, c8 = st.columns(4)
c5.metric("Profit factor", "" if pd.isna(metrics.get("profit_factor", np.nan)) else f"{metrics.get('profit_factor'):.2f}")
c6.metric("Expectancy", fmt_pct(metrics.get("expectancy", 0)))
c7.metric("Avg holding days", fmt_num(metrics.get("avg_holding_days", 0)))
c8.metric("Avg recommended holding days", fmt_num(metrics.get("avg_recommended_holding_days", 0)))

# =========================================================
# Tabs
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Actual vs Potential",
    "Stock detail",
    "Exit quality",
    "Performance metrics",
    "Market / Setup framework",
    "Raw data"
])

with tab1:
    st.subheader("Actual vs Potential P&L by stock")

    stock_summary = (
        analysis.groupby("ticker", as_index=False)
        .agg(
            trades=("ticker", "count"),
            quantity=("quantity", "sum"),
            actual_pnl=("actual_pnl", "sum"),
            potential_pnl=("potential_pnl", "sum"),
            missed_pnl=("potential_minus_actual", "sum"),
            avg_capture_rate=("capture_rate", "mean"),
            avg_holding_days=("holding_days", "mean"),
            avg_recommended_holding_days=("recommended_holding_days", "mean"),
            avg_increase_after_sell_pct=("increase_after_sell_pct", "mean"),
            avg_decrease_after_sell_pct=("decrease_after_sell_pct", "mean"),
        )
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stock_summary["ticker"],
        y=stock_summary["actual_pnl"],
        name="Actual P&L"
    ))
    fig.add_trace(go.Bar(
        x=stock_summary["ticker"],
        y=stock_summary["potential_pnl"],
        name="Potential P&L"
    ))
    fig.update_layout(
        barmode="group",
        title="Actual vs Potential P&L by stock",
        yaxis_title="VND",
        xaxis_title="Ticker",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(style_money_cols(stock_summary), use_container_width=True)

with tab2:
    st.subheader("Stock detail: price path, buy/sell points, and peak")

    selected_ticker = st.selectbox("Select stock", sorted(analysis["ticker"].dropna().unique()))
    stock_trades = analysis[analysis["ticker"] == selected_ticker].sort_values("sell_date")
    stock_prices = prices[prices["ticker"] == selected_ticker].sort_values("date") if not prices.empty else pd.DataFrame()

    if stock_prices.empty:
        st.warning("No price history for this stock.")
    else:
        min_plot_date = stock_trades["buy_date"].min()
        if pd.isna(min_plot_date):
            min_plot_date = stock_trades["sell_date"].min() - timedelta(days=30)
        max_plot_date = stock_trades["sell_date"].max() + timedelta(days=evaluation_days)
        plot_prices = stock_prices[
            (stock_prices["date"] >= min_plot_date) &
            (stock_prices["date"] <= max_plot_date)
        ]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=plot_prices["date"],
            y=plot_prices["close"],
            mode="lines",
            name="Close price"
        ))

        buy_points = stock_trades.dropna(subset=["buy_date", "buy_price"])
        sell_points = stock_trades.dropna(subset=["sell_date", "sell_price"])
        peak_points = stock_trades.dropna(subset=["peak_date", "peak_price"])

        fig2.add_trace(go.Scatter(
            x=buy_points["buy_date"],
            y=buy_points["buy_price"],
            mode="markers",
            name="Buy point",
            marker=dict(size=10, symbol="triangle-up")
        ))
        fig2.add_trace(go.Scatter(
            x=sell_points["sell_date"],
            y=sell_points["sell_price"],
            mode="markers",
            name="Actual sell point",
            marker=dict(size=10, symbol="circle")
        ))
        fig2.add_trace(go.Scatter(
            x=peak_points["peak_date"],
            y=peak_points["peak_price"],
            mode="markers",
            name="Recommended/peak sell point",
            marker=dict(size=11, symbol="star")
        ))

        fig2.update_layout(
            title=f"{selected_ticker}: price path with buy/sell/peak points",
            xaxis_title="Date",
            yaxis_title="Price",
            height=550,
        )
        st.plotly_chart(fig2, use_container_width=True)

    detail_cols = [
        "ticker", "buy_date", "sell_date", "quantity", "buy_price", "sell_price",
        "actual_pnl", "potential_pnl", "potential_minus_actual", "capture_rate",
        "holding_days", "recommended_holding_days", "days_to_peak_after_sell",
        "increase_after_sell_pct", "decrease_after_sell_pct"
    ]
    st.dataframe(style_money_cols(stock_trades[detail_cols]), use_container_width=True)

with tab3:
    st.subheader("Exit quality")

    exit_cols = [
        "ticker", "buy_date", "sell_date", "quantity",
        "actual_pnl", "potential_pnl", "potential_minus_actual",
        "capture_rate", "increase_after_sell_pct", "decrease_after_sell_pct",
        "holding_days", "recommended_holding_days", "days_to_peak_after_sell",
        "peak_price", "peak_date"
    ]

    top_missed = analysis.sort_values("potential_minus_actual", ascending=False).head(15)
    top_best = analysis.sort_values("potential_minus_actual", ascending=True).head(15)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Top missed opportunities")
        st.dataframe(style_money_cols(top_missed[exit_cols]), use_container_width=True)
    with right:
        st.markdown("#### Best exits / least missed")
        st.dataframe(style_money_cols(top_best[exit_cols]), use_container_width=True)

    st.markdown("#### Highest increase after selling")
    inc = analysis.sort_values("increase_after_sell_pct", ascending=False).head(15)
    st.dataframe(style_money_cols(inc[exit_cols]), use_container_width=True)

    st.markdown("#### Biggest decrease after selling")
    dec = analysis.sort_values("decrease_after_sell_pct", ascending=True).head(15)
    st.dataframe(style_money_cols(dec[exit_cols]), use_container_width=True)

with tab4:
    st.subheader("Performance metrics")

    valid = analysis.dropna(subset=["actual_pnl"]).copy()
    valid["is_win"] = valid["actual_pnl"] > 0

    st.markdown("#### Key metrics")
    metrics_table = pd.DataFrame([{
        "actual_pnl": metrics.get("total_actual_pnl"),
        "potential_pnl": metrics.get("total_potential_pnl"),
        "missed_pnl": metrics.get("total_missed_pnl"),
        "win_rate": metrics.get("win_rate"),
        "avg_win_return": metrics.get("avg_win"),
        "avg_loss_return": metrics.get("avg_loss"),
        "expectancy": metrics.get("expectancy"),
        "profit_factor": metrics.get("profit_factor"),
        "max_drawdown": metrics.get("max_drawdown"),
        "consecutive_losses": metrics.get("consecutive_losses"),
    }])
    st.dataframe(style_money_cols(metrics_table), use_container_width=True)

    st.markdown("#### Equity curve from realized trades")
    equity = valid.sort_values("sell_date").copy()
    equity["cumulative_pnl"] = equity["actual_pnl"].cumsum()
    fig3 = px.line(equity, x="sell_date", y="cumulative_pnl", title="Cumulative realized P&L")
    st.plotly_chart(fig3, use_container_width=True)

    optional_dims = [c for c in ["setup", "market_condition", "entry_trigger", "behavior_tag"] if c in trades.columns]
    if optional_dims:
        st.markdown("#### Win rate by optional labels")
        dim = st.selectbox("Dimension", optional_dims)
        # Merge optional labels approximately by ticker/date/side if available
        label_source = trades[trades["side"] == "SELL"][["ticker", "date", dim]].dropna()
        tmp = valid.merge(label_source, left_on=["ticker", "sell_date"], right_on=["ticker", "date"], how="left")
        by_dim = tmp.groupby(dim, as_index=False).agg(
            trades=("actual_pnl", "count"),
            win_rate=("is_win", "mean"),
            actual_pnl=("actual_pnl", "sum"),
            profit_factor=("actual_pnl", lambda s: s[s > 0].sum() / abs(s[s < 0].sum()) if abs(s[s < 0].sum()) > 0 else np.nan)
        )
        st.dataframe(style_money_cols(by_dim), use_container_width=True)
    else:
        st.info("For setup/market-condition analysis, add columns like `setup`, `market_condition`, `entry_trigger`, `behavior_tag` to your trade file later.")

with tab5:
    st.subheader("Market / Setup framework")

    st.markdown("""
This dashboard is ready for the 4-level trading analysis framework:

1. **Market Context**: trend/range, gap up/down, volume vs average, VN30/VNINDEX moving average.
2. **Setup Analysis**: breakout, pullback, reversal, counter-trend, futures scalp.
3. **Performance Metrics**: expectancy, profit factor, max drawdown, consecutive losses.
4. **Exit Analysis**: actual vs potential P&L, capture rate, days to peak, missed profit.

Current file supports the strongest first layer:
- Actual vs Potential P&L
- Days to peak after sale
- Holding days vs recommended holding days
- Capture rate
- Win rate / expectancy / profit factor

To unlock market-condition analysis, add optional columns:
`setup`, `market_condition`, `entry_trigger`, `stop_loss`, `target`, `behavior_tag`.
""")

with tab6:
    st.subheader("Raw standardized orders")
    st.dataframe(style_money_cols(trades), use_container_width=True)

    st.subheader("FIFO realized trade lots")
    st.dataframe(style_money_cols(analysis), use_container_width=True)

    csv = analysis.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download analyzed trades CSV",
        data=csv,
        file_name="analyzed_trades_fifo_exit_quality.csv",
        mime="text/csv"
    )
