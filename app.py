import streamlit as st
import pandas as pd
import altair as alt
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Holdings por Portafolio", layout="wide")
load_dotenv()
mysql_connection_string = os.getenv("MYSQL_CONNECTION_STRING")
engine = create_engine(mysql_connection_string)

@st.cache_data
def load_data():
    query = """
    SELECT portfolio_id, date, holding, allocation
    FROM Holdings
    WHERE date >= CURDATE() - INTERVAL 7 DAY
    """
    df = pd.read_sql(query, con=engine)
    df['date'] = pd.to_datetime(df['date'])
    return df
    
def METRICS():
    # Define la consulta SQL para seleccionar las columnas deseadas
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
    
    # Cargar los datos usando pandas
    df = pd.read_sql(query, con=engine)

    # Convertir las columnas de fechas a datetime
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    # Convertir las columnas numéricas a los tipos adecuados si es necesario
    numeric_columns = [
        'average_daily_value', 'median_daily_value', 'max_daily_value', 'min_daily_value', 
        'cumulative_return_percent', 'cagr_percent', 'year_to_date_percent', 'last_year_percent',
        'two_years_percent', 'hit_rate_percent', 'value_at_risk_var', 'conditional_var_cvar', 
        'sharpe_ratio', 'sortino_ratio', 'max_drawdown_percent', 'volatility_ann_percent', 
        'calmar_ratio', 'skew', 'kurtosis', 'recovery_factor', 'sp500_cumulative_return_percent', 
        'treynor_index', 'beta', 'alpha', 'risk_parity', 'mean_drawdown_depth', 
        'maximum_drawdown_recovery_time', 'omega_ratio', 'ulcer_index', 'tail_ratio', 'gain_to_pain_ratio'
    ]
    
    # Convertir columnas numéricas a tipo float
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return df



df = load_data()
try:
    metrics_summary_df = METRICS()  # Cambié esta línea
except Exception as e:
    st.error(f"❌ Error cargando tabla MetricsSummary: {e}")
    st.stop()



st.sidebar.title("Filtros")

min_date = datetime.today() - timedelta(days=7)
max_date = datetime.today()
date_selected = st.sidebar.date_input("Selecciona una fecha", max_value=max_date, min_value=min_date)

available_portfolios = df[df["date"] == pd.to_datetime(date_selected)]["portfolio_id"].unique()
portfolio_selected = st.sidebar.selectbox("Selecciona un portafolio", available_portfolios)

filtered_df = df[
    (df["date"] == pd.to_datetime(date_selected)) &
    (df["portfolio_id"] == portfolio_selected)
]
# Filtro por portfolio_id
filtered_summary = metrics_summary_df[
    metrics_summary_df["portfolio_id"] == portfolio_selected
]

st.title("Gráfica de Holdings por Portafolio")
st.write(f"📅 Fecha seleccionada: {date_selected.strftime('%d/%m/%Y')}")
st.write(f"📁 Portafolio: `{portfolio_selected}`")
st.dataframe(filtered_df)

if not filtered_df.empty:
    chart = alt.Chart(filtered_df).mark_arc().encode(
        theta=alt.Theta(field="allocation", type="quantitative"),
        color=alt.Color(field="holding", type="nominal"),
        tooltip=["holding", "allocation"]
    ).properties(
        title=f"Holdings para {portfolio_selected} el {date_selected.strftime('%d/%m/%Y')}"
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning("No hay datos para la fecha y portafolio seleccionados.")

st.subheader("📊 Métricas")

if not filtered_summary.empty:
    st.dataframe(filtered_summary)
else:
    st.info("No hay métricas para el portafolio seleccionado.")
