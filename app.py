import streamlit as st
import pandas as pd
import altair as alt
import requests
import re
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from io import StringIO
import boto3
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Mi Dashboard", layout="wide")
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# A) CONFIGURACIÓN DEL MOTOR SQL Y CARGA DE DATOS CACHEADOS
# ─────────────────────────────────────────────────────────────────────────────
MYSQL_CONNECTION_STRING = st.secrets["MYSQL_CONNECTION_STRING"]

engine = create_engine(MYSQL_CONNECTION_STRING)


@st.cache_data(show_spinner=False)
def load_holdings_last_7d() -> pd.DataFrame:
    """
    Carga las participaciones (Holdings) de los últimos 7 días y las convierte a datetime.
    """
    query = """
    SELECT portfolio_id, date, holding, allocation
    FROM Holdings
    WHERE date >= CURDATE() - INTERVAL 7 DAY
    """
    df = pd.read_sql(query, con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """
    Carga la tabla Metrics completa, convierte fechas y columnas numéricas.
    """
    query = """
    SELECT 
        id,
        portfolio_id,
        start_date,
        end_date,
        average_daily_value,
        median_daily_value,
        max_daily_value,
        min_daily_value,
        cumulative_return_percent,
        cagr_percent,
        year_to_date_percent,
        last_year_percent,
        two_years_percent,
        hit_rate_percent,
        value_at_risk_var,
        conditional_var_cvar,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown_percent,
        volatility_ann_percent,
        calmar_ratio,
        skew,
        kurtosis,
        recovery_factor,
        sp500_cumulative_return_percent,
        treynor_index,
        beta,
        alpha,
        risk_parity,
        mean_drawdown_depth,
        maximum_drawdown_recovery_time,
        omega_ratio,
        ulcer_index,
        tail_ratio,
        gain_to_pain_ratio
    FROM Metrics
    """
    df = pd.read_sql(query, con=engine)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    numeric_cols = [
        "average_daily_value", "median_daily_value", "max_daily_value", "min_daily_value",
        "cumulative_return_percent", "cagr_percent", "year_to_date_percent", "last_year_percent",
        "two_years_percent", "hit_rate_percent", "value_at_risk_var", "conditional_var_cvar",
        "sharpe_ratio", "sortino_ratio", "max_drawdown_percent", "volatility_ann_percent",
        "calmar_ratio", "skew", "kurtosis", "recovery_factor", "sp500_cumulative_return_percent",
        "treynor_index", "beta", "alpha", "risk_parity", "mean_drawdown_depth",
        "maximum_drawdown_recovery_time", "omega_ratio", "ulcer_index", "tail_ratio", "gain_to_pain_ratio"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_all_holdings() -> pd.DataFrame:
    """
    Carga la tabla completa de Holdings (útil para las tres páginas).
    """
    df = pd.read_sql("SELECT * FROM Holdings", con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B) FINVIZ NEWS SCRAPER (CÓDIGO SIN CAMBIOS)
# ─────────────────────────────────────────────────────────────────────────────

WHITELIST_DOMAINS = [
    "businesswire.com",
    "seekingalpha.com",
    "investopedia.com",
    "reuters.com",
    "prnewswire.com",
    "globenewswire.com",
    "youtube.com",
    "investors.com",
    "barrons.com",
    "qz.com",
    "wsj.com",
]

BLACKLIST_KEYWORDS = [
    "Top Stock Movers",
    "Gains and Losses",
    "Stock Picks",
    "Mad Dash",
    "breaks out",
    "traders expect",
    "Here’s Where Traders",
    "Top Stock Movers Now",
    "Gains and Losses Today",
    "Analyst",
    "Analysts",
    "Stocks Pop",
    "Stocks Slide",
    "ETF",
    "Magnificent Seven",
    "Rally",
    "Stocks Rise",
    "Stock Rises",
]

BLACKLIST_REGEX = [
    r"(?i)\bTop\b.*?\bStocks\b",
    r"(?i)\bStocks\s+\S+\s+Moving\b",
]


def is_whitelisted_domain(url: str) -> bool:
    for domain in WHITELIST_DOMAINS:
        if domain in url:
            return True
    return False


def contains_blacklisted_phrase(text: str) -> bool:
    lower = text.lower()
    for phrase in BLACKLIST_KEYWORDS:
        if phrase.lower() in lower:
            return True
    for pattern in BLACKLIST_REGEX:
        if re.search(pattern, text):
            return True
    return False
def get_finviz_news(ticker: str) -> list[dict]:
    import requests
    from bs4 import BeautifulSoup

    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")

    news_table = soup.find("table", class_="fullview-news-outer")
    if not news_table:
        return []

    filtered_news = []
    for row in news_table.find_all("tr"):
        a_tag = row.find("a")
        td_tag = row.find("td")

        if not a_tag or not td_tag:
            continue

        link = a_tag.get("href", "")
        headline = a_tag.text.strip()
        time_data = td_tag.text.strip()

        if (
            link.startswith("https://finance.yahoo.com/")
            or link.startswith("https://www.marketwatch.com/")
            or link.startswith("/news/")
        ):
            continue

        if not is_whitelisted_domain(link):
            continue

        if contains_blacklisted_phrase(headline):
            continue

        filtered_news.append(
            {
                "ticker": ticker,
                "datetime": time_data,
                "title": headline,
                "link": link,
            }
        )

    return filtered_news


def get_top_news_for_tickers(tickers: list[str], max_total: int = 15) -> list[dict]:
    all_news = {t: get_finviz_news(t) for t in tickers}
    combined = []
    idx = 0
    while len(combined) < max_total:
        added = False
        for t in tickers:
            articles = all_news.get(t, [])
            if idx < len(articles):
                combined.append(articles[idx])
                added = True
                if len(combined) >= max_total:
                    break
        if not added:
            break
        idx += 1
    return combined


def get_latest_buy_symbols(df_holdings: pd.DataFrame, portfolio_id: str) -> list[str]:
    df_pf = df_holdings[df_holdings["portfolio_id"] == portfolio_id].copy()
    if df_pf.empty:
        return []
    df_pf = df_pf.sort_values("date")
    date_groups = df_pf.groupby("date")["allocation"]
    date_stats = (
        date_groups.agg(
            [("n_unique_allocs", "nunique"), ("first_alloc", "first")]
        )
        .reset_index()
    )
    date_stats["is_uniform"] = date_stats["n_unique_allocs"] == 1
    date_stats["prev_uniform"] = date_stats["is_uniform"].shift(1).fillna(False)
    buy_indices = [
        idx for idx, row in date_stats.iterrows()
        if row["is_uniform"] and not row["prev_uniform"]
    ]
    if not buy_indices:
        return []
    last_buy_idx = buy_indices[-1]
    buy_date = date_stats.loc[last_buy_idx, "date"]
    symbols = df_pf[df_pf["date"] == buy_date]["holding"].tolist()
    return symbols


def get_portfolio_latest_news(
    df_holdings: pd.DataFrame, portfolio_id: str, max_total: int = 15
) -> list[dict]:
    tickers = get_latest_buy_symbols(df_holdings, portfolio_id)
    if not tickers:
        return []
    top_news = get_top_news_for_tickers(tickers, max_total=max_total)
    return top_news


# ─────────────────────────────────────────────────────────────────────────────
# C) S3 HELPERS + “REBALANCE” SNAPSHOTS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_s3_csv(s3_client, bucket: str, file_key: str) -> pd.DataFrame:
    """
    Descarga un CSV de S3 y lo devuelve como pandas.DataFrame.
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        csv_data = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_data))
        return df
    except Exception as e:
        st.error(f"Error cargando {file_key} desde S3: {e}")
        return None


def get_all_adjusted_close(
    updated_holdings_data: pd.DataFrame,
    updated_etfs_data: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame):
    """
    A partir de los DataFrames “_Adjusted_close”, genera dos tablas:
      - holdings_prices: índice = Date, columnas = símbolos de acciones
      - etfs_prices:     índice = Date, columnas = símbolos de ETFs
    """
    holdings_cols = [
        col for col in updated_holdings_data.columns
        if col.endswith("_Adjusted_close")
    ]
    etfs_cols = [
        col for col in updated_etfs_data.columns
        if col.endswith("_Adjusted_close")
    ]

    holdings_data = updated_holdings_data[["Date"] + holdings_cols].dropna(how="all")
    etfs_data     = updated_etfs_data[["Date"] + etfs_cols].dropna(how="all")

    holdings_data["Date"] = pd.to_datetime(holdings_data["Date"])
    etfs_data["Date"]     = pd.to_datetime(etfs_data["Date"])

    holdings_data = holdings_data.rename(
        columns={col: col.replace("_Adjusted_close", "") for col in holdings_cols}
    )
    etfs_data = etfs_data.rename(
        columns={col: col.replace("_Adjusted_close", "") for col in etfs_cols}
    )

    holdings_prices = holdings_data.set_index("Date").sort_index()
    etfs_prices     = etfs_data.set_index("Date").sort_index()

    return holdings_prices, etfs_prices


def build_all_rebalance_snapshots_from_s3(
    df_holdings: pd.DataFrame,
    portfolio_id: str,
    holdings_prices: pd.DataFrame,
    etfs_prices: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Para un portfolio_id, encuentra todos los “Buy Date → After Date” 
    donde las asignaciones fueron uniformes, luego calcula:
      - “Price Bought” y “Price After” para cada símbolo,
      - Retorno porcentual.
    Devuelve un DataFrame con columnas:
      Buy Date, After Date, Symbol, Allocation, Price Bought, Price After, Return %.
    """
    # 1) Filtrar full holdings history al portfolio_id
    df_pf = df_holdings[df_holdings["portfolio_id"] == portfolio_id].copy()
    if df_pf.empty:
        raise ValueError(f"No hay datos para portfolio_id = '{portfolio_id}'")

    df_pf["date"] = pd.to_datetime(df_pf["date"])
    df_pf = df_pf.sort_values("date")

    # 2) Armamos date_stats
    date_groups = df_pf.groupby("date")["allocation"]
    date_stats = (
        date_groups.agg([("n_unique_allocs", "nunique"), ("first_alloc", "first")])
        .reset_index()
    )
    date_stats["is_uniform"] = date_stats["n_unique_allocs"] == 1
    date_stats["uniform_alloc"] = date_stats.apply(
        lambda r: r["first_alloc"] if r["is_uniform"] else pd.NA, axis=1
    )
    date_stats = date_stats.sort_values("date").reset_index(drop=True)

    # 3) ID de buy_indices (primer día donde is_uniform=True y anterior no lo era)
    date_stats["prev_is_uniform"] = date_stats["is_uniform"].shift(1).fillna(False)
    buy_indices = [
        idx for idx, row in date_stats.iterrows()
        if row["is_uniform"] and not row["prev_is_uniform"]
    ]
    if not buy_indices:
        raise ValueError(f"No se encontraron bloques uniformes para '{portfolio_id}'")

    # 4) Construir pares (buy_date, after_date, uniform_alloc)
    blocks = []
    for i, buy_idx in enumerate(buy_indices):
        buy_date = date_stats.loc[buy_idx, "date"]
        uniform_alloc = float(date_stats.loc[buy_idx, "uniform_alloc"])

        if i + 1 < len(buy_indices):
            next_buy_idx = buy_indices[i + 1]
            after_date = date_stats.loc[next_buy_idx, "date"]
        else:
            # Último bloque: After Date = última fecha disponible
            after_date = date_stats["date"].iloc[-1]

        if buy_date < after_date:
            blocks.append((buy_date, after_date, uniform_alloc))

    # 5) Para cada bloque, buscar precios y calcular Retorno
    all_rows = []
    for buy_date, after_date, uniform_alloc in blocks:
        symbols_on_buy = df_pf[df_pf["date"] == buy_date]["holding"].tolist()
        if not symbols_on_buy:
            continue

        for sym in symbols_on_buy:
            # Precio en buy_date
            try:
                price_buy = holdings_prices.loc[buy_date, sym]
            except KeyError:
                if etfs_prices is not None and sym in etfs_prices.columns:
                    price_buy = etfs_prices.loc[buy_date, sym]
                else:
                    raise KeyError(f"No se encontró precio para {sym} en {buy_date.date()}")

            # Precio en after_date
            try:
                price_after = holdings_prices.loc[after_date, sym]
            except KeyError:
                if etfs_prices is not None and sym in etfs_prices.columns:
                    price_after = etfs_prices.loc[after_date, sym]
                else:
                    raise KeyError(f"No se encontró precio para {sym} en {after_date.date()}")

            ret_decimal = (price_after - price_buy) / price_buy

            all_rows.append({
                "Buy Date":     buy_date.date(),
                "After Date":   after_date.date(),
                "Symbol":       sym,
                "Allocation":   uniform_alloc,
                "Price Bought": price_buy,
                "Price After":  price_after,
                "Return %":     round(ret_decimal * 100, 2)
            })

    df_all_snapshots = pd.DataFrame(all_rows)
    df_all_snapshots = df_all_snapshots.sort_values(
        ["Buy Date", "Symbol"]
    ).reset_index(drop=True)
    return df_all_snapshots


@st.cache_data(show_spinner=False)
def load_price_lookup_from_s3() -> (pd.DataFrame, pd.DataFrame):
    """
    1) Crea el cliente S3 según las credenciales en .env
    2) Descarga updated_holdings_data.csv y updated_etfs_data.csv
    3) Llama a get_all_adjusted_close(...) y retorna (holdings_prices, etfs_prices).
    """
    # 1) Leer credenciales de .env
    aws_access_key = st.secrets["aws"]["aws_access_key"]
    aws_secret_key = st.secrets["aws"]["aws_secret_key"]
    aws_region = st.secrets["aws"]["aws_region"]
    bucket_name = st.secrets["aws"]["bucket_name"]
    
    if not all([aws_access_key, aws_secret_key, bucket_name]):
        st.error("Faltan credenciales de AWS o nombre de bucket en .env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET).")
        return None, None

    # 2) Crear cliente S3
    s3_client = boto3.client(
        "s3",
        aws_access_key_id     = aws_access_key,
        aws_secret_access_key = aws_secret_key,
        region_name           = aws_region,
    )

    # 3) Descargar CSVs
    updated_holdings_data = fetch_s3_csv(s3_client, bucket_name, "updated_holdings_data.csv")
    updated_etfs_data     = fetch_s3_csv(s3_client, bucket_name, "updated_etfs_data.csv")

    if updated_holdings_data is None or updated_etfs_data is None:
        return None, None

    # 4) Generar diccionarios de precios
    holdings_prices, etfs_prices = get_all_adjusted_close(
        updated_holdings_data,
        updated_etfs_data
    )
    return holdings_prices, etfs_prices


# ─────────────────────────────────────────────────────────────────────────────
# D) PÁGINA 1: Portfolio Monitor
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_monitor_page():
    st.header("📊 Portfolio Monitor")
    st.write("En esta página puedes ver las participaciones (holdings) de un portafolio, junto con sus métricas históricas.")

    df = load_holdings_last_7d()    # Holdings últimos 7 días
    metrics_df = load_metrics()     # Tabla Metrics

    st.sidebar.markdown("### Filtros – Portfolio Monitor")
    min_date = datetime.today() - timedelta(days=7)
    max_date = datetime.today()
    date_selected = st.sidebar.date_input(
        "Selecciona una fecha",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    ts_selected = pd.to_datetime(date_selected)
    available_portfolios = df[df["date"] == ts_selected]["portfolio_id"].unique()
    if len(available_portfolios) == 0:
        st.warning("No hay portafolios con datos para la fecha seleccionada.")
        return

    portfolio_selected = st.sidebar.selectbox(
        "Selecciona un portafolio",
        options=available_portfolios
    )

    filtered_df = df[
        (df["date"] == ts_selected) &
        (df["portfolio_id"] == portfolio_selected)
    ]
    filtered_metrics = metrics_df[metrics_df["portfolio_id"] == portfolio_selected]

    st.title(f"Holdings de `{portfolio_selected}` – {date_selected.strftime('%d/%m/%Y')}")
    if filtered_df.empty:
        st.warning("No hay participaciones (holdings) para la fecha y portafolio seleccionados.")
    else:
        st.dataframe(filtered_df, use_container_width=True)
        chart = (
            alt.Chart(filtered_df)
            .mark_arc()
            .encode(
                theta=alt.Theta(field="allocation", type="quantitative"),
                color=alt.Color(field="holding", type="nominal"),
                tooltip=["holding", "allocation"]
            )
            .properties(
                title=f"Allocations para `{portfolio_selected}` el {date_selected.strftime('%d/%m/%Y')}"
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("📈 Métricas Históricas")
    if filtered_metrics.empty:
        st.info("No hay métricas registradas para este portafolio.")
    else:
        st.dataframe(filtered_metrics, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# E) PÁGINA 2: Portfolio News
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_news_page():
    st.header("📰 Portfolio News")
    st.write(
        "Esta página obtiene las últimas noticias desde Finviz para los holdings "
        "de la última Buy Date de un portafolio."
    )

    all_holdings = load_all_holdings()

    st.sidebar.markdown("### Filtros – Portfolio News")
    all_portfolios = all_holdings["portfolio_id"].unique().tolist()
    if not all_portfolios:
        st.warning("Aún no se han cargado portafolios en la base de datos.")
        return

    chosen_portfolio = st.sidebar.selectbox(
        "Selecciona un portafolio para ver sus noticias",
        options=sorted(all_portfolios)
    )

    if st.sidebar.button("🔍 Obtener Noticias"):
        with st.spinner("Buscando noticias..."):
            top15 = get_portfolio_latest_news(all_holdings, chosen_portfolio, max_total=15)

        if not top15:
            st.info("No se encontraron noticias para ese portafolio (o no hay Buy Date).")
            return

        news_df = pd.DataFrame(top15)
        news_df["link"] = news_df["link"].apply(lambda url: f"[Ver artículo]({url})")

        st.subheader(f"🎯 Noticias Destacadas para `{chosen_portfolio}`")
        st.dataframe(
            news_df[["ticker", "datetime", "title", "link"]],
            use_container_width=True
        )
    else:
        st.info("Presiona 'Obtener Noticias' para webscrape en Finviz.")


# ─────────────────────────────────────────────────────────────────────────────
# F) PÁGINA 3: Portfolio Rebalance (“Rebalance Periods”)
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_rebalance_page():
    st.header("🔄 Portfolio Rebalance")
    st.write(
        "En esta sección puedes ver todos los “Periodos de rebalanceo” de un portafolio "
        "(Buy Date → After Date), los precios de compra/venta y el retorno de cada símbolo."
    )

    # 1) Cargar full holdings (histórico) para detectar Buy Dates
    all_holdings = load_all_holdings()

    # 2) Cargar precios desde S3 (cacheado)
    holdings_prices, etfs_prices = load_price_lookup_from_s3()
    if holdings_prices is None or etfs_prices is None:
        st.stop()  # Si falló la carga de precios, detenemos aquí.

    st.sidebar.markdown("### Filtros – Portfolio Rebalance")
    # dropdown con TODOS los portfolio_id disponibles
    all_portfolios = all_holdings["portfolio_id"].unique().tolist()
    if not all_portfolios:
        st.warning("No hay portafolios en la base de datos.")
        return

    chosen_portfolio = st.sidebar.selectbox(
        "Selecciona un portafolio para Rebalance Periods",
        options=sorted(all_portfolios)
    )

    if st.sidebar.button("▶️ Cargar Rebalance Periods"):
        with st.spinner("Calculando Rebalance Periods..."):
            try:
                df_rebalances = build_all_rebalance_snapshots_from_s3(
                    df_holdings=all_holdings,
                    portfolio_id=chosen_portfolio,
                    holdings_prices=holdings_prices,
                    etfs_prices=etfs_prices
                )
            except Exception as e:
                st.error(f"Error al calcular rebalance periods: {e}")
                return

        if df_rebalances.empty:
            st.info("No se encontraron bloques de asignación uniforme para este portafolio.")
            return

        st.subheader(f"🎯 Rebalance Periods para `{chosen_portfolio}`")

        # 3) Extraer todas las combinaciones únicas (Buy Date, After Date)
        unique_periods = (
            df_rebalances[["Buy Date", "After Date"]]
            .drop_duplicates()
            .sort_values(["Buy Date", "After Date"])
            .reset_index(drop=True)
        )

        # 4) Para cada período, mostramos un pequeño DataFrame
        for idx, row in unique_periods.iterrows():
            buy_dt = row["Buy Date"]
            after_dt = row["After Date"]
            st.markdown(f"**Periodo {idx+1}:** {buy_dt} → {after_dt}")
            period_df = df_rebalances[
                (df_rebalances["Buy Date"] == buy_dt) &
                (df_rebalances["After Date"] == after_dt)
            ].copy()

            # Mostramos las columnas más relevantes para analizar: Symbol, Allocation, Price Bought, Price After, Return %
            st.dataframe(
                period_df[[
                    "Symbol", "Allocation", "Price Bought", "Price After", "Return %"
                ]].reset_index(drop=True),
                use_container_width=True
            )

            # Agregar un separador entre períodos
            st.markdown("---")

    else:
        st.info("Presiona 'Cargar Rebalance Periods' para ver los detalles.")


# ─────────────────────────────────────────────────────────────────────────────
# G) ENTRYPOINT: MENÚ LATERAL PARA CAMBIAR DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Navegación:")
page = st.sidebar.radio(
    "",
    ("Portfolio Monitor", "Portfolio News", "Portfolio Rebalance")
)

if page == "Portfolio Monitor":
    portfolio_monitor_page()
elif page == "Portfolio News":
    portfolio_news_page()
elif page == "Portfolio Rebalance":
    portfolio_rebalance_page()
