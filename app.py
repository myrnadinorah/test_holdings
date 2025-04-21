import streamlit as st
import pandas as pd
import altair as alt
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# Configuración de la app
st.set_page_config(page_title="Holdings por Portafolio", layout="wide")

# Parámetros de conexión
host = "portfolio-database.cluster-c7b6x1wx8cfw.us-east-1.rds.amazonaws.com"
port = 3306
db = "DB_MOMENTO"
user = "admimomento"
password = "MomentoIyY2025"

# Crear motor de conexión
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}")

# Cargar datos una sola vez
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

df = load_data()

# Sidebar para selección de fecha y portfolio_id
st.sidebar.title("Filtros")

# Selección de fecha (solo una)
min_date = datetime.today() - timedelta(days=7)
max_date = datetime.today()
date_selected = st.sidebar.date_input("Selecciona una fecha", max_value=max_date, min_value=min_date)

# Filtrar portafolios disponibles
available_portfolios = df[df["date"] == pd.to_datetime(date_selected)]["portfolio_id"].unique()
portfolio_selected = st.sidebar.selectbox("Selecciona un portafolio", available_portfolios)

# Filtrar el DataFrame
filtered_df = df[
    (df["date"] == pd.to_datetime(date_selected)) &
    (df["portfolio_id"] == portfolio_selected)
]

# Mostrar título y tabla
st.title("Gráfica de Holdings por Portafolio")
st.write(f"📅 Fecha seleccionada: {date_selected.strftime('%d/%m/%Y')}")
st.write(f"📁 Portafolio: `{portfolio_selected}`")
st.dataframe(filtered_df)

# Gráfica de pastel
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
