
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
# Heatmap / decision-support helpers
# =========================================================

HEATMAP_COLORS = {
    "very_positive": "background-color: #0f5132; color: white;",
    "positive": "background-color: #d1e7dd; color: #0f5132;",
    "neutral": "background-color: #fff3cd; color: #664d03;",
    "negative": "background-color: #f8d7da; color: #842029;",
    "very_negative": "background-color: #842029; color: white;",
}

HIGHER_IS_BETTER_HINTS = [
    "growth", "roe", "roa", "margin", "liquidity", "value", "volume", "breadth",
    "eps", "revenue", "profit", "cash", "net_buy", "foreign_buy", "above_ma",
    "ma20_ma60", "price_vs_ma", "rsi", "momentum", "return", "upside", "yield"
]
LOWER_IS_BETTER_HINTS = [
    "pe", "p/e", "pb", "p/b", "ps", "p/s", "ev_ebitda", "debt", "leverage",
    "drawdown", "usd_vnd", "dxy", "foreign_sell", "risk", "volatility", "margin_pressure"
]
PERCENTILE_HINTS = ["percentile", "pctile", "rank_5y", "rank_3y"]


def infer_direction(column_name):
    cl = str(column_name).lower()
    if any(k in cl for k in LOWER_IS_BETTER_HINTS):
        return "lower_better"
    if any(k in cl for k in HIGHER_IS_BETTER_HINTS):
        return "higher_better"
    return "higher_better"


def status_from_value(value, column_name="", direction=None):
    if pd.isna(value):
        return ""
    try:
        v = float(value)
    except Exception:
        return ""

    cl = str(column_name).lower()
    direction = direction or infer_direction(column_name)

    # Percentile columns are usually 0-1 or 0-100. For valuation/risk lower is better.
    if any(k in cl for k in PERCENTILE_HINTS):
        if abs(v) <= 1:
            v = v * 100
        lower_better = direction == "lower_better" or any(k in cl for k in LOWER_IS_BETTER_HINTS)
        if lower_better:
            if v <= 20: return "very_positive"
            if v <= 40: return "positive"
            if v <= 60: return "neutral"
            if v <= 80: return "negative"
            return "very_negative"
        else:
            if v >= 80: return "very_positive"
            if v >= 60: return "positive"
            if v >= 40: return "neutral"
            if v >= 20: return "negative"
            return "very_negative"

    # Ratio columns around 1.0 such as MA20/MA60.
    if "ma20_ma60" in cl or "ma20/ma60" in cl or "ratio" in cl:
        if v >= 1.15: return "very_positive"
        if v >= 1.03: return "positive"
        if v >= 0.97: return "neutral"
        if v >= 0.85: return "negative"
        return "very_negative"

    # Percentage-like values may be represented as decimals or whole percentages.
    is_pct_like = any(k in cl for k in ["growth", "change", "return", "yield", "margin", "roe", "roa", "vs_ma", "pct", "%"])
    if is_pct_like and abs(v) <= 1:
        v = v * 100

    if direction == "lower_better":
        if v <= -10: return "very_positive"
        if v <= 0: return "positive"
        if v <= 10: return "neutral"
        if v <= 25: return "negative"
        return "very_negative"

    if v >= 25: return "very_positive"
    if v >= 5: return "positive"
    if v >= -5: return "neutral"
    if v >= -20: return "negative"
    return "very_negative"


def heatmap_style_numeric(df, exclude_cols=None, direction_overrides=None):
    exclude_cols = set(exclude_cols or [])
    direction_overrides = direction_overrides or {}

    def style_cell(v, col):
        if col in exclude_cols or not pd.api.types.is_numeric_dtype(df[col]):
            return ""
        status = status_from_value(v, col, direction_overrides.get(col))
        return HEATMAP_COLORS.get(status, "")

    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in df.columns:
        if col in exclude_cols or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        styles[col] = df[col].map(lambda v, c=col: style_cell(v, c))
    return styles


def format_decision_df(df):
    fmt = {}
    for c in df.columns:
        cl = str(c).lower()
        if pd.api.types.is_numeric_dtype(df[c]):
            if any(k in cl for k in ["pct", "percentile", "growth", "return", "margin", "roe", "roa", "yield", "vs_ma", "change", "rate"]):
                fmt[c] = lambda x: "" if pd.isna(x) else (f"{x*100:.1f}%" if abs(float(x)) <= 1 else f"{float(x):.1f}%")
            elif any(k in cl for k in ["price", "value", "volume", "liquidity", "margin_outstanding", "cash", "revenue", "profit"]):
                fmt[c] = lambda x: "" if pd.isna(x) else f"{float(x):,.0f}".replace(",", ".")
            else:
                fmt[c] = lambda x: "" if pd.isna(x) else f"{float(x):,.2f}".rstrip("0").rstrip(".")
    return fmt


def render_heatmap_table(df, caption=None, exclude_cols=None, direction_overrides=None):
    if df is None or df.empty:
        st.info("No data to display.")
        return
    if caption:
        st.caption(caption)
    styled = df.style.apply(
        lambda x: heatmap_style_numeric(df, exclude_cols=exclude_cols, direction_overrides=direction_overrides),
        axis=None
    ).format(format_decision_df(df))
    st.dataframe(styled, use_container_width=True)


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


def normalize_flexible_table(df, ticker_required=False, date_col_optional=True):
    """Normalize optional CSV/XLSX inputs while preserving user-defined columns.

    Useful for:
    - market_context.csv: metric-level or date-level market indicators
    - watchlist.csv: stock candidates not yet bought
    - portfolio.csv: current holdings, if user wants to upload it later
    """
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if ticker_required and "ticker" not in df.columns:
        st.error("File này cần có cột `ticker`.")
        st.stop()

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    for c in df.columns:
        cl = c.lower()
        if cl in ["date", "as_of_date", "report_date"]:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            continue
        if c not in ["ticker", "sector", "industry", "note", "status", "cycle", "trigger", "reason", "watch_reason"]:
            converted = pd.to_numeric(df[c], errors="ignore")
            df[c] = converted
    return df


def latest_price_snapshot(prices):
    if prices is None or prices.empty:
        return pd.DataFrame()
    p = prices.sort_values(["ticker", "date"]).copy()
    p["ma20"] = p.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    p["ma60"] = p.groupby("ticker")["close"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    p["ma200"] = p.groupby("ticker")["close"].transform(lambda s: s.rolling(200, min_periods=60).mean())
    p["return_20d"] = p.groupby("ticker")["close"].pct_change(20)
    p["return_60d"] = p.groupby("ticker")["close"].pct_change(60)
    p["price_vs_ma20"] = p["close"] / p["ma20"] - 1
    p["price_vs_ma60"] = p["close"] / p["ma60"] - 1
    p["price_vs_ma200"] = p["close"] / p["ma200"] - 1
    snap = p.groupby("ticker", as_index=False).tail(1)
    return snap[["ticker", "date", "close", "ma20", "ma60", "ma200", "return_20d", "return_60d", "price_vs_ma20", "price_vs_ma60", "price_vs_ma200"]]


def build_watchlist_view(watchlist, prices):
    if watchlist is None or watchlist.empty:
        return pd.DataFrame()
    out = watchlist.copy()
    snap = latest_price_snapshot(prices)
    if not snap.empty:
        out = out.merge(snap, on="ticker", how="left", suffixes=("", "_market"))

    if "target_price" in out.columns and "close" in out.columns:
        out["upside_to_target"] = out["target_price"] / out["close"] - 1
    if "fair_value" in out.columns and "close" in out.columns:
        out["upside_to_fair_value"] = out["fair_value"] / out["close"] - 1

    trigger_cols = [c for c in out.columns if c.startswith("trigger_")]
    if trigger_cols:
        def readiness(row):
            vals = []
            for c in trigger_cols:
                v = row.get(c)
                if isinstance(v, str):
                    vals.append(v.strip().lower() in ["yes", "y", "true", "1", "done", "pass", "ready", "đạt"])
                else:
                    vals.append(bool(v) if not pd.isna(v) else False)
            return f"{sum(vals)}/{len(vals)}"
        out["trigger_readiness"] = out.apply(readiness, axis=1)
    return out


def build_missed_opportunity_view(watchlist, prices):
    """Track stocks that were watched but not bought from their added_date.

    Required/optional watchlist columns:
    - ticker: required
    - added_date / watch_date: optional. If absent, the first available price date is used.
    - intended_capital: optional. Used to estimate missed opportunity value.
    - reason_not_bought / watch_reason / note: optional qualitative tags.
    """
    if watchlist is None or watchlist.empty or prices is None or prices.empty:
        return pd.DataFrame()

    date_col = None
    for c in ["added_date", "watch_date", "start_date", "candidate_date"]:
        if c in watchlist.columns:
            date_col = c
            break

    rows = []
    price_df = prices.sort_values(["ticker", "date"]).copy()
    for _, row in watchlist.iterrows():
        ticker = row.get("ticker")
        tp = price_df[price_df["ticker"] == ticker].copy()
        if tp.empty:
            continue

        added_date = pd.to_datetime(row.get(date_col), errors="coerce") if date_col else pd.NaT
        if pd.isna(added_date):
            added_date = tp["date"].min()

        after = tp[tp["date"] >= added_date].sort_values("date")
        if after.empty:
            continue

        start = after.iloc[0]
        latest = after.iloc[-1]
        peak = after.loc[after["close"].idxmax()]
        trough = after.loc[after["close"].idxmin()]

        current_return = latest["close"] / start["close"] - 1 if start["close"] else np.nan
        max_return = peak["close"] / start["close"] - 1 if start["close"] else np.nan
        max_drawdown_from_watch = trough["close"] / start["close"] - 1 if start["close"] else np.nan

        intended_capital = row.get("intended_capital", np.nan)
        try:
            intended_capital = float(intended_capital)
        except Exception:
            intended_capital = np.nan

        out = {
            "ticker": ticker,
            "added_date": added_date,
            "start_price": start["close"],
            "latest_date": latest["date"],
            "latest_price": latest["close"],
            "current_return_since_added": current_return,
            "peak_price_since_added": peak["close"],
            "peak_date_since_added": peak["date"],
            "max_return_since_added": max_return,
            "max_drawdown_since_added": max_drawdown_from_watch,
            "missed_profit_est": intended_capital * current_return if pd.notna(intended_capital) and pd.notna(current_return) else np.nan,
        }
        for c in ["sector", "industry", "cycle", "reason_not_bought", "watch_reason", "note"]:
            if c in watchlist.columns:
                out[c] = row.get(c)
        rows.append(out)

    return pd.DataFrame(rows)


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

    st.header("Optional decision-support files")
    st.caption("Add raw indicators. The dashboard will show the real numbers and color them as a heatmap; no black-box scores.")
    market_context_file = st.file_uploader(
        "Upload market_context.csv/xlsx",
        type=["csv", "xlsx", "xls"],
        help="Suggested columns: metric,current,3m_ago,yoy,percentile_5y,trend,note. You can also use date-level rows."
    )
    watchlist_file = st.file_uploader(
        "Upload watchlist.csv/xlsx",
        type=["csv", "xlsx", "xls"],
        help="Required column: ticker. Optional: sector, pe, pb, roe, eps_growth_yoy, revenue_growth_yoy, target_price, fair_value, trigger_* columns."
    )
    portfolio_file = st.file_uploader(
        "Upload current_portfolio.csv/xlsx",
        type=["csv", "xlsx", "xls"],
        help="Optional. Suggested columns: ticker, quantity, avg_price, market_value, sector."
    )

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

market_context = pd.DataFrame()
watchlist = pd.DataFrame()
portfolio = pd.DataFrame()
watchlist_view = pd.DataFrame()
missed_watchlist = pd.DataFrame()

# Read optional files before fetching prices so the price loader includes
# tickers that are not in historical trades yet, especially watchlist names.
if market_context_file is not None:
    market_context = normalize_flexible_table(read_uploaded_file(market_context_file), ticker_required=False)

if watchlist_file is not None:
    watchlist = normalize_flexible_table(read_uploaded_file(watchlist_file), ticker_required=True)

if portfolio_file is not None:
    portfolio = normalize_flexible_table(read_uploaded_file(portfolio_file), ticker_required=True)

all_tickers = set(trades["ticker"].dropna().unique())
if not watchlist.empty and "ticker" in watchlist.columns:
    all_tickers.update(watchlist["ticker"].dropna().unique())
if not portfolio.empty and "ticker" in portfolio.columns:
    all_tickers.update(portfolio["ticker"].dropna().unique())
tickers = sorted(all_tickers)

min_date_candidates = [trades["date"].min() - timedelta(days=10)]
for df_optional in [watchlist, portfolio]:
    for dc in ["added_date", "watch_date", "start_date", "candidate_date", "as_of_date", "date"]:
        if not df_optional.empty and dc in df_optional.columns:
            dmin = pd.to_datetime(df_optional[dc], errors="coerce").min()
            if pd.notna(dmin):
                min_date_candidates.append(dmin - timedelta(days=10))

min_date = min(min_date_candidates)
max_date = trades["date"].max() + timedelta(days=evaluation_days + 10)

prices = pd.DataFrame()
if price_mode == "Auto Yahoo Finance":
    with st.spinner("Fetching historical prices for trades + watchlist + portfolio from Yahoo Finance..."):
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

if not watchlist.empty:
    watchlist_view = build_watchlist_view(watchlist, prices)
    missed_watchlist = build_missed_opportunity_view(watchlist, prices)

if not portfolio.empty:
    snap = latest_price_snapshot(prices)
    if not snap.empty:
        portfolio = portfolio.merge(snap, on="ticker", how="left", suffixes=("", "_market"))
    if "quantity" in portfolio.columns and "avg_price" in portfolio.columns and "close" in portfolio.columns:
        portfolio["market_value_est"] = portfolio["quantity"] * portfolio["close"]
        portfolio["unrealized_pnl_est"] = (portfolio["close"] - portfolio["avg_price"]) * portfolio["quantity"]
        portfolio["unrealized_return_est"] = portfolio["close"] / portfolio["avg_price"] - 1

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Actual vs Potential",
    "Stock detail",
    "Exit quality",
    "Performance metrics",
    "Market / Setup framework",
    "Market heatmap",
    "Watchlist",
    "Portfolio / Exposure",
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

    st.markdown("#### Setup × Regime Matrix")
    st.caption("Shows which setups work better under each market condition/regime, using actual raw performance metrics rather than a score.")
    setup_col = "setup" if "setup" in trades.columns else None
    regime_col = None
    for c in ["market_regime", "market_condition", "regime"]:
        if c in trades.columns:
            regime_col = c
            break

    if setup_col and regime_col:
        label_cols = ["ticker", "date", setup_col, regime_col]
        sell_labels = trades[trades["side"] == "SELL"][label_cols].dropna(subset=[setup_col, regime_col])
        matrix_src = valid.merge(sell_labels, left_on=["ticker", "sell_date"], right_on=["ticker", "date"], how="left")
        matrix_src = matrix_src.dropna(subset=[setup_col, regime_col])

        if matrix_src.empty:
            st.info("No matched setup/regime labels found on SELL rows yet.")
        else:
            setup_regime = matrix_src.groupby([setup_col, regime_col], as_index=False).agg(
                trades=("actual_pnl", "count"),
                win_rate=("is_win", "mean"),
                actual_pnl=("actual_pnl", "sum"),
                avg_return=("actual_return", "mean"),
                profit_factor=("actual_pnl", lambda s: s[s > 0].sum() / abs(s[s < 0].sum()) if abs(s[s < 0].sum()) > 0 else np.nan),
                avg_holding_days=("holding_days", "mean"),
            )
            render_heatmap_table(
                setup_regime,
                caption="Green/red is applied directly to win rate, P&L, return, profit factor, and holding metrics.",
                exclude_cols=[setup_col, regime_col],
            )

            pivot_metric = st.selectbox(
                "Matrix metric",
                ["win_rate", "actual_pnl", "avg_return", "profit_factor", "trades"],
                key="setup_regime_metric"
            )
            pivot = setup_regime.pivot(index=setup_col, columns=regime_col, values=pivot_metric)
            st.dataframe(
                pivot.style.background_gradient(axis=None).format(format_decision_df(pivot)),
                use_container_width=True
            )
    else:
        st.info("To unlock Setup × Regime Matrix, add `setup` plus `market_condition` or `market_regime` to your order history file.")

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
    st.subheader("Market heatmap: raw indicators, no composite score")
    st.caption("Upload `market_context.csv/xlsx` to track liquidity, valuation, earnings, FX, foreign flow, futures basis, or any market regime indicators. Numeric cells are colored directly from the actual value / percentile / trend.")

    if market_context.empty:
        st.info("No market context file uploaded yet.")
        st.markdown("""
Suggested format:

| metric | current | 3m_ago | yoy | percentile_5y | trend | note |
|---|---:|---:|---:|---:|---|---|
| hose_value_bn | 28500 | 22100 | 0.32 | 0.78 | up | Liquidity improving |
| vnindex_pe | 12.8 | 11.9 | -0.05 | 0.22 | neutral | Valuation still reasonable |
| usd_vnd_change_1m | 0.012 | 0.008 | 0.03 | 0.65 | up | FX pressure rising |

Rules are intentionally simple: lower percentile is green for valuation/risk columns; higher growth/liquidity/breadth columns are green.
""")
    else:
        non_numeric = [c for c in market_context.columns if not pd.api.types.is_numeric_dtype(market_context[c])]
        render_heatmap_table(
            market_context,
            caption="Heatmap is applied only to numeric columns. Text columns such as metric/trend/note remain unchanged.",
            exclude_cols=non_numeric,
        )

        numeric_cols = [c for c in market_context.columns if pd.api.types.is_numeric_dtype(market_context[c])]
        metric_col = "metric" if "metric" in market_context.columns else None
        if metric_col and numeric_cols:
            selected_metric_value_col = st.selectbox("Chart market metric column", numeric_cols, key="market_metric_col")
            chart_df = market_context[[metric_col, selected_metric_value_col]].dropna().copy()
            if not chart_df.empty:
                fig_mkt = px.bar(chart_df, x=metric_col, y=selected_metric_value_col, title=f"{selected_metric_value_col} by market indicator")
                st.plotly_chart(fig_mkt, use_container_width=True)

with tab7:
    st.subheader("Watchlist: stocks not yet bought")
    st.caption("This section is designed for candidates you are tracking but have not bought. It keeps actual raw metrics and uses heatmap coloring instead of scoring.")

    if watchlist_view.empty:
        st.info("No watchlist file uploaded yet.")
        st.markdown("""
Suggested format:

| ticker | sector | pe | pb | roe | eps_growth_yoy | revenue_growth_yoy | target_price | fair_value | trigger_breakout_ma200 | trigger_eps_recovery | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| HPG | Steel | 10.5 | 1.4 | 0.18 | 0.45 | 0.28 | 36000 | 34000 | yes | yes | Early recovery candidate |
| FPT | Tech | 22.0 | 5.1 | 0.24 | 0.21 | 0.18 | 145000 | 140000 | no | yes | Quality compounder |

The app will automatically add latest close, MA20/60/200, returns, price-vs-MA, and upside columns when price history is available.
""")
    else:
        display_cols = list(watchlist_view.columns)
        preferred = [
            "ticker", "sector", "industry", "cycle", "close", "target_price", "fair_value",
            "upside_to_target", "upside_to_fair_value", "pe", "pb", "roe",
            "eps_growth_yoy", "revenue_growth_yoy", "return_20d", "return_60d",
            "price_vs_ma20", "price_vs_ma60", "price_vs_ma200", "trigger_readiness", "note"
        ]
        ordered = [c for c in preferred if c in display_cols] + [c for c in display_cols if c not in preferred]
        wl_display = watchlist_view[ordered]
        non_numeric = [c for c in wl_display.columns if not pd.api.types.is_numeric_dtype(wl_display[c])]
        render_heatmap_table(wl_display, exclude_cols=non_numeric)

        if "sector" in watchlist_view.columns:
            st.markdown("#### Watchlist sector count")
            sector_count = watchlist_view.groupby("sector", as_index=False).agg(stocks=("ticker", "count"))
            fig_sector = px.bar(sector_count, x="sector", y="stocks", title="Number of candidates by sector")
            st.plotly_chart(fig_sector, use_container_width=True)

        if "upside_to_target" in watchlist_view.columns:
            st.markdown("#### Upside to target")
            top_upside = watchlist_view.dropna(subset=["upside_to_target"]).sort_values("upside_to_target", ascending=False).head(20)
            if not top_upside.empty:
                fig_up = px.bar(top_upside, x="ticker", y="upside_to_target", title="Watchlist upside to target")
                fig_up.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_up, use_container_width=True)

        st.markdown("#### Missed Opportunity Tracker")
        st.caption("Tracks candidates from `added_date` / `watch_date` until the latest available price. Add `intended_capital` to estimate missed profit in VND.")
        if missed_watchlist.empty:
            st.info("To activate this, add `added_date` or `watch_date` to watchlist.csv and make sure price history covers those tickers.")
        else:
            missed_cols = list(missed_watchlist.columns)
            preferred_missed = [
                "ticker", "sector", "cycle", "added_date", "start_price", "latest_price",
                "current_return_since_added", "peak_price_since_added", "max_return_since_added",
                "max_drawdown_since_added", "missed_profit_est", "reason_not_bought", "watch_reason", "note"
            ]
            missed_ordered = [c for c in preferred_missed if c in missed_cols] + [c for c in missed_cols if c not in preferred_missed]
            missed_display = missed_watchlist[missed_ordered].sort_values("current_return_since_added", ascending=False)
            non_numeric_missed = [c for c in missed_display.columns if not pd.api.types.is_numeric_dtype(missed_display[c])]
            render_heatmap_table(missed_display, exclude_cols=non_numeric_missed)

            if "current_return_since_added" in missed_watchlist.columns:
                top_missed_chart = missed_watchlist.dropna(subset=["current_return_since_added"]).sort_values("current_return_since_added", ascending=False).head(20)
                if not top_missed_chart.empty:
                    fig_missed = px.bar(
                        top_missed_chart,
                        x="ticker",
                        y="current_return_since_added",
                        title="Watchlist return since added / watched"
                    )
                    fig_missed.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig_missed, use_container_width=True)

        csv_wl = watchlist_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download enriched watchlist CSV",
            data=csv_wl,
            file_name="enriched_watchlist_heatmap.csv",
            mime="text/csv"
        )

with tab8:
    st.subheader("Portfolio / Exposure")
    st.caption("Optional current holdings view. This does not replace the trade journal; it helps compare what you own vs what you are watching.")

    if portfolio.empty:
        st.info("No current portfolio file uploaded yet.")
        st.markdown("""
Suggested format:

| ticker | quantity | avg_price | sector | note |
|---|---:|---:|---|---|
| FPT | 100 | 120000 | Tech | Core position |
| HPG | 1000 | 28000 | Steel | Cyclical recovery |
""")
    else:
        non_numeric = [c for c in portfolio.columns if not pd.api.types.is_numeric_dtype(portfolio[c])]
        render_heatmap_table(portfolio, exclude_cols=non_numeric)

        if "sector" in portfolio.columns and "market_value_est" in portfolio.columns:
            exposure = portfolio.groupby("sector", as_index=False).agg(market_value_est=("market_value_est", "sum"))
            fig_exp = px.pie(exposure, names="sector", values="market_value_est", title="Estimated exposure by sector")
            st.plotly_chart(fig_exp, use_container_width=True)
        elif "market_value" in portfolio.columns and "sector" in portfolio.columns:
            exposure = portfolio.groupby("sector", as_index=False).agg(market_value=("market_value", "sum"))
            fig_exp = px.pie(exposure, names="sector", values="market_value", title="Exposure by sector")
            st.plotly_chart(fig_exp, use_container_width=True)

with tab9:
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
