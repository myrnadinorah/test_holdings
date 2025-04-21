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
    
@st.cache_data
def load_metrics():
    query = "SELECT * FROM MetricsSummary"
    df = pd.read_sql(query, con=engine)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    return df
    
df = load_data()
metrics_summary_df = load_metrics_summary()

# Filtro por portfolio_id
filtered_summary = metrics_summary_df[
    metrics_summary_df["portfolio_id"] == portfolio_selected
]

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
