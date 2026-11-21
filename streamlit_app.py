"""
Пазарен радар — Streamlit приложение за следене на пазарни трендове
и консенсуса на анализаторите.

Стартиране:
    pip install -r requirements.txt
    streamlit run app.py

ВАЖНО: Сигналите са информативни и не са инвестиционен съвет.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Пазарен радар", page_icon="📈", layout="wide")

# ---------------------------------------------------------------
# 1. Активи по подразбиране (може да добавяш свои в страничното меню)
# ---------------------------------------------------------------
DEFAULT_ASSETS = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Световни акции (VT)": "VT",
    "Европейски акции (VGK)": "VGK",
    "Злато (GLD)": "GLD",
    "Облигации САЩ 20+ г. (TLT)": "TLT",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Bitcoin": "BTC-USD",
}


# ---------------------------------------------------------------
# 2. Зареждане на данни (кеширано, за да не се тегли всеки път)
# ---------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):  # новите версии на yfinance
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


@st.cache_data(ttl=6 * 3600)
def load_analysts(ticker: str) -> dict:
    """Консенсус на анализаторите, ценови цели и новини.
    За ETF и крипто обикновено няма препоръки — тогава връща празно."""
    t = yf.Ticker(ticker)
    out = {"summary": None, "targets": None, "news": []}
    try:
        out["summary"] = t.recommendations_summary
    except Exception:
        pass
    try:
        out["targets"] = t.analyst_price_targets
    except Exception:
        pass
    try:
        out["news"] = (t.news or [])[:5]
    except Exception:
        pass
    return out


# ---------------------------------------------------------------
# 3. Индикатори за тренд
# ---------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def trend_metrics(df: pd.DataFrame) -> dict:
    c = df["Close"].dropna()
    last = float(c.iloc[-1])

    def ret(days):
        return (last / float(c.iloc[-days - 1]) - 1) * 100 if len(c) > days else np.nan

    daily = c.pct_change().dropna()
    return {
        "Цена": last,
        "SMA50": float(c.rolling(50).mean().iloc[-1]),
        "SMA200": float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else np.nan,
        "RSI": float(rsi(c).iloc[-1]),
        "1М %": ret(21),
        "3М %": ret(63),
        "1Г %": ret(252),
        "Волатилност %": float(daily[-252:].std() * np.sqrt(252) * 100),
        "Макс. спад %": float(((c / c.cummax()) - 1).min() * 100),
    }


def trend_score(m: dict) -> float:
    """Проста точкова система от -3 до +3."""
    s = 0.0
    if not np.isnan(m["SMA200"]):
        s += 1 if m["Цена"] > m["SMA200"] else -1          # дългосрочен тренд
        s += 1 if m["SMA50"] > m["SMA200"] else -1         # "златен"/"смъртен" кръст
    if not np.isnan(m["3М %"]):
        s += 1 if m["3М %"] > 0 else -1                    # инерция
    if m["RSI"] > 70:
        s -= 0.5                                           # прегрят
    elif m["RSI"] < 30:
        s += 0.5                                           # препродаден
    return s


def analyst_score(summary) -> float | None:
    """От -2 (силно продавай) до +2 (силно купувай)."""
    if summary is None or len(summary) == 0:
        return None
    row = summary.iloc[0]  # текущият месец ("0m")
    sb, b, h, s, ss = (row.get(k, 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
    total = sb + b + h + s + ss
    if total == 0:
        return None
    return (2 * sb + b - s - 2 * ss) / total


def verdict(ts: float, an: float | None) -> str:
    trend = "възходящ" if ts >= 1.5 else "низходящ" if ts <= -1.5 else "неясен"
    if an is None:
        return f"Тренд: {trend} · няма данни от анализатори"
    mood = "оптимисти" if an >= 0.8 else "песимисти" if an <= -0.2 else "неутрални"
    return f"Тренд: {trend} · анализаторите са {mood}"


# ---------------------------------------------------------------
# 4. Интерфейс
# ---------------------------------------------------------------
st.title("📈 Пазарен радар")
st.caption("Информативен инструмент. Не е инвестиционен съвет. Данните са от Yahoo Finance и може да са със закъснение.")

with st.sidebar:
    st.header("Настройки")
    chosen = st.multiselect("Активи", list(DEFAULT_ASSETS), default=list(DEFAULT_ASSETS)[:6])
    custom = st.text_input("Добави тикери (със запетая)", placeholder="напр. AMZN, ASML, ETH-USD")
    period = st.selectbox("Период", ["1y", "2y", "5y"], index=1)

tickers = {name: DEFAULT_ASSETS[name] for name in chosen}
for t in [x.strip().upper() for x in custom.split(",") if x.strip()]:
    tickers[t] = t

if not tickers:
    st.info("Избери поне един актив от страничното меню.")
    st.stop()

tab_overview, tab_detail, tab_help = st.tabs(["Преглед", "Детайли", "Как да чета сигналите"])

# --- Преглед: таблица с всички активи ---
rows, cache = [], {}
with st.spinner("Зареждам данни..."):
    for name, t in tickers.items():
        df = load_prices(t, period)
        if df.empty or len(df) < 60:
            st.warning(f"Няма достатъчно данни за {t}.")
            continue
        m = trend_metrics(df)
        an = analyst_score(load_analysts(t)["summary"])
        ts = trend_score(m)
        cache[name] = (t, df, m)
        rows.append({"Актив": name, **{k: m[k] for k in ["Цена", "RSI", "1М %", "3М %", "1Г %", "Волатилност %", "Макс. спад %"]},
                     "Тренд (−3…+3)": ts, "Анализатори (−2…+2)": an, "Сигнал": verdict(ts, an)})

with tab_overview:
    if rows:
        table = pd.DataFrame(rows).sort_values("Тренд (−3…+3)", ascending=False)
        st.dataframe(
            table.style.format(precision=2, na_rep="—")
                 .background_gradient(subset=["Тренд (−3…+3)"], cmap="RdYlGn", vmin=-3, vmax=3),
            use_container_width=True, hide_index=True,
        )

# --- Детайли: графика, метрики, анализатори, новини ---
with tab_detail:
    if cache:
        pick = st.selectbox("Актив", list(cache))
        t, df, m = cache[pick]
        c = df["Close"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c.index, y=c, name="Цена"))
        fig.add_trace(go.Scatter(x=c.index, y=c.rolling(50).mean(), name="SMA50", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=c.index, y=c.rolling(200).mean(), name="SMA200", line=dict(dash="dash")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(4)
        cols[0].metric("Цена", f"{m['Цена']:.2f}", f"{m['1М %']:.1f}% за 1М")
        cols[1].metric("RSI (14)", f"{m['RSI']:.0f}")
        cols[2].metric("Волатилност", f"{m['Волатилност %']:.1f}%")
        cols[3].metric("Макс. спад", f"{m['Макс. спад %']:.1f}%")

        a = load_analysts(t)
        st.subheader("Мнение на анализаторите")
        if a["summary"] is not None and len(a["summary"]):
            st.bar_chart(a["summary"].iloc[0][["strongBuy", "buy", "hold", "sell", "strongSell"]])
        else:
            st.write("Няма препоръки (типично за ETF, суровини и крипто).")
        if isinstance(a["targets"], dict) and a["targets"].get("mean"):
            tg = a["targets"]
            upside = (tg["mean"] / m["Цена"] - 1) * 100
            st.write(f"Средна ценова цел: **{tg['mean']:.2f}** (диапазон {tg.get('low', 0):.2f} – {tg.get('high', 0):.2f}), "
                     f"т.е. {upside:+.1f}% спрямо текущата цена.")

        st.subheader("Последни новини")
        for item in a["news"]:
            content = item.get("content", item)  # yfinance смени формата през 2025
            title = content.get("title", "")
            url = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
            source = (content.get("provider") or {}).get("displayName") or item.get("publisher", "")
            if title:
                st.markdown(f"- [{title}]({url}) — *{source}*")

with tab_help:
    st.markdown("""
**Тренд (−3…+3)** събира четири проверки:
1. Цената над 200-дневната средна? → +1, иначе −1
2. 50-дневната средна над 200-дневната? → +1, иначе −1
3. Положителна доходност за 3 месеца? → +1, иначе −1
4. RSI над 70 (прегрят) → −0.5; под 30 (препродаден) → +0.5

**Анализатори (−2…+2)** е средна оценка от препоръките: силно купувай = +2, купувай = +1,
задръж = 0, продавай = −1, силно продавай = −2.

**Ограничения:** трендовете описват миналото и не гарантират бъдещето; анализаторите
исторически са склонни към оптимизъм; високата волатилност и големият макс. спад
показват колко болезнено може да падне активът.
""")
