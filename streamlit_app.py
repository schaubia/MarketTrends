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
ASSET_GROUPS = {
    "Индекси САЩ": {
        "S&P 500 (SPY)": "SPY",
        "Nasdaq 100 (QQQ)": "QQQ",
        "Dow Jones (DIA)": "DIA",
        "Малки компании (IWM)": "IWM",
    },
    "Световни пазари": {
        "Целият свят (VT)": "VT",
        "Европа (VGK)": "VGK",
        "Развиващи се пазари (EEM)": "EEM",
        "Япония (EWJ)": "EWJ",
        "Китай (FXI)": "FXI",
        "Индия (INDA)": "INDA",
    },
    "Сектори САЩ": {
        "Технологии (XLK)": "XLK",
        "Финанси (XLF)": "XLF",
        "Енергетика (XLE)": "XLE",
        "Здравеопазване (XLV)": "XLV",
        "Индустрия (XLI)": "XLI",
        "Потребителски стоки (XLY)": "XLY",
        "Основни стоки (XLP)": "XLP",
        "Комунални услуги (XLU)": "XLU",
        "Недвижими имоти (XLRE)": "XLRE",
        "Материали (XLB)": "XLB",
        "Комуникации (XLC)": "XLC",
    },
    "Теми и мегатрендове": {
        "Полупроводници (SMH)": "SMH",
        "Роботика и ИИ (BOTZ)": "BOTZ",
        "Киберсигурност (CIBR)": "CIBR",
        "Чиста енергия (ICLN)": "ICLN",
        "Отбрана и авиация (ITA)": "ITA",
        "Уран (URA)": "URA",
        "Литий и батерии (LIT)": "LIT",
    },
    "Големи технологични акции": {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Alphabet": "GOOGL",
        "Amazon": "AMZN",
        "Meta": "META",
        "Tesla": "TSLA",
    },
    "Европейски акции": {
        "ASML": "ASML",
        "SAP": "SAP",
        "Novo Nordisk": "NVO",
    },
    "Облигации": {
        "САЩ 20+ г. (TLT)": "TLT",
        "САЩ 7-10 г. (IEF)": "IEF",
        "САЩ 1-3 г. (SHY)": "SHY",
        "Корпоративни (LQD)": "LQD",
        "Високодоходни (HYG)": "HYG",
    },
    "Суровини": {
        "Злато (GLD)": "GLD",
        "Сребро (SLV)": "SLV",
        "Петрол (USO)": "USO",
        "Мед (CPER)": "CPER",
        "Земеделие (DBA)": "DBA",
    },
    "Крипто": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",
    },
    "Дивидентни ETF": {
        "Растящи дивиденти (VIG)": "VIG",
        "Висок дивидент (VYM)": "VYM",
        "Качествени дивиденти (SCHD)": "SCHD",
        "Дивидентни аристократи (NOBL)": "NOBL",
        "Международни дивиденти (VYMI)": "VYMI",
    },
    "Дивидентни акции": {
        "Coca-Cola": "KO",
        "PepsiCo": "PEP",
        "Johnson & Johnson": "JNJ",
        "Procter & Gamble": "PG",
        "McDonald's": "MCD",
        "AbbVie": "ABBV",
        "Realty Income": "O",
        "Verizon": "VZ",
        "Exxon Mobil": "XOM",
    },
    "Банки": {
        "Банки САЩ (KBE)": "KBE",
        "Регионални банки САЩ (KRE)": "KRE",
        "Европейски банки (EUFN)": "EUFN",
        "JPMorgan Chase": "JPM",
        "Bank of America": "BAC",
        "Goldman Sachs": "GS",
        "Morgan Stanley": "MS",
        "HSBC": "HSBC",
        "Santander": "SAN",
        "ING": "ING",
        "UBS": "UBS",
    },
}
# Само за тези категории има смисъл да питаме за препоръки на анализатори
STOCK_CATEGORIES = {"Големи технологични акции", "Европейски акции", "Дивидентни акции", "Банки"}

# Плоски речници за удобство: име → тикер, име → категория
DEFAULT_ASSETS = {n: t for g in ASSET_GROUPS.values() for n, t in g.items()}
CATEGORY_OF = {n: cat for cat, g in ASSET_GROUPS.items() for n in g}


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


@st.cache_data(ttl=24 * 3600)
def dividend_yield(ticker: str, price: float) -> float:
    """Дивидентна доходност за последните 12 месеца, в %.
    Пример: KO е платила 4 × 0.51 = 2.04 $ за година при цена 70 $ → 2.9%.
    Връща 0 за активи без дивиденти (напр. NVDA почти 0, крипто 0)."""
    try:
        d = yf.Ticker(ticker).dividends
    except Exception:
        return np.nan
    if d is None or d.empty:
        return 0.0
    since = pd.Timestamp.now(tz=d.index.tz) - pd.Timedelta(days=365)
    return float(d[d.index >= since].sum()) / price * 100


@st.cache_data(ttl=24 * 3600)
def guess_category(ticker: str) -> str:
    """Автоматична категория за тикери, добавени ръчно.
    Пример: 'AMZN' → 'Свои: акция · Consumer Cyclical', 'ETH-USD' → 'Свои: крипто'."""
    if ticker.endswith("-USD"):
        return "Свои: крипто"
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return "Свои: друго"
    kind = info.get("quoteType", "")
    if kind == "EQUITY":
        return f"Свои: акция · {info.get('sector', 'без сектор')}"
    if kind == "ETF":
        return f"Свои: ETF · {info.get('category') or 'без категория'}"
    return "Свои: друго"


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
    cats = st.multiselect("Категории", list(ASSET_GROUPS),
                          default=["Индекси САЩ", "Сектори САЩ", "Суровини"])
    pool = [n for c in cats for n in ASSET_GROUPS[c]]
    chosen = st.multiselect("Активи (махни тези, които не искаш)", pool, default=pool)
    custom = st.text_input("Добави тикери (със запетая)", placeholder="напр. KO, JPM, VNQ, DOGE-USD")
    period = st.selectbox("Период", ["1y", "2y", "5y"], index=1)
    st.caption(f"Общо налични: {len(DEFAULT_ASSETS)} актива в {len(ASSET_GROUPS)} категории. "
               "Много активи = по-бавно първо зареждане (после се кешира).")

tickers = {name: DEFAULT_ASSETS[name] for name in chosen}
for t in [x.strip().upper() for x in custom.split(",") if x.strip()]:
    tickers[t] = t
    CATEGORY_OF[t] = guess_category(t)

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
        cat = CATEGORY_OF.get(name, "Свои: друго")
        wants_analysts = cat in STOCK_CATEGORIES or cat.startswith("Свои: акция")
        an = analyst_score(load_analysts(t)["summary"]) if wants_analysts else None
        ts = trend_score(m)
        cache[name] = (t, df, m)
        divy = 0.0 if t.endswith("-USD") else dividend_yield(t, m["Цена"])
        rows.append({"Категория": cat, "Актив": name, "Цена": m["Цена"], "Дивидент %": divy,
                     **{k: m[k] for k in ["RSI", "1М %", "3М %", "1Г %", "Волатилност %", "Макс. спад %"]},
                     "Тренд (−3…+3)": ts, "Анализатори (−2…+2)": an, "Сигнал": verdict(ts, an)})

with tab_overview:
    if rows:
        table = pd.DataFrame(rows).sort_values("Тренд (−3…+3)", ascending=False)
        table.insert(2, "Клас", pd.cut(table["Тренд (−3…+3)"], bins=[-10, -1.5, 1.4, 10],
                                       labels=["🔴 Низходящ", "🟡 Неясен", "🟢 Възходящ"]).astype(str))

        # --- Обобщение по категории: кои части на пазара вървят нагоре ---
        st.subheader("По категории")
        by_cat = (table.groupby("Категория")
                       .agg(Брой=("Актив", "count"),
                            Среден_тренд=("Тренд (−3…+3)", "mean"),
                            Средно_3М=("3М %", "mean"),
                            Среден_дивидент=("Дивидент %", "mean"),
                            Възходящи=("Клас", lambda s: (s == "🟢 Възходящ").sum()))
                       .sort_values("Среден_тренд", ascending=False)
                       .reset_index())
        c1, c2 = st.columns([3, 2])
        c1.dataframe(by_cat.style.format(precision=2), use_container_width=True, hide_index=True)
        fig_cat = go.Figure(go.Bar(x=by_cat["Среден_тренд"], y=by_cat["Категория"], orientation="h",
                                   marker_color=["#2e9e5b" if v > 0 else "#d64545" for v in by_cat["Среден_тренд"]]))
        fig_cat.update_layout(height=60 + 30 * len(by_cat), margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(autorange="reversed"), xaxis_title="Среден тренд")
        c2.plotly_chart(fig_cat, use_container_width=True)

        # --- Филтър по клас и пълната таблица ---
        st.subheader("Всички активи")
        show = st.multiselect("Покажи класове", ["🟢 Възходящ", "🟡 Неясен", "🔴 Низходящ"],
                              default=["🟢 Възходящ", "🟡 Неясен", "🔴 Низходящ"])
        table = table[table["Клас"].isin(show)]
        st.dataframe(
            table.style.format(precision=2, na_rep="—")
                 .background_gradient(subset=["Тренд (−3…+3)"], cmap="RdYlGn", vmin=-3, vmax=3)
                 .background_gradient(subset=["Дивидент %"], cmap="Blues", vmin=0, vmax=8),
            use_container_width=True, hide_index=True,
        )
        st.caption("Дивидент % = изплатените дивиденти за последните 12 месеца ÷ текущата цена. "
                   "Кликни върху заглавие на колона, за да сортираш по нея.")

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

**Дивидент %** показва колко пари в брой носи активът годишно спрямо цената му.
Пример: при 2.9% и инвестиция 1000 $ получаваш около 29 $ годишно (преди данъци).
Много висок дивидент (над 7–8%) понякога е предупреждение: цената е паднала, защото
пазарът очаква дивидентът да бъде намален.

**Ограничения:** трендовете описват миналото и не гарантират бъдещето; анализаторите
исторически са склонни към оптимизъм; високата волатилност и големият макс. спад
показват колко болезнено може да падне активът.
""")
