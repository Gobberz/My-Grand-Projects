# pages/1_Retention.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retention Analysis", layout="wide")
st.title("Retention Analysis")

st.markdown("Загрузите файл с колонками: `user_id`, `install_date`, `event_date`")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["install_date", "event_date"])

    period_type = st.selectbox("Группировать по", ["День", "Неделя", "Месяц"])

    if period_type == "День":
        df['install_period'] = df['install_date'].dt.to_period('D').apply(lambda r: r.start_time)
        df['event_period'] = df['event_date'].dt.to_period('D').apply(lambda r: r.start_time)
        period_label = 'день'
    elif period_type == "Неделя":
        df['install_period'] = df['install_date'].dt.to_period('W').apply(lambda r: r.start_time)
        df['event_period'] = df['event_date'].dt.to_period('W').apply(lambda r: r.start_time)
        period_label = 'неделю'
    else:
        df['install_period'] = df['install_date'].dt.to_period('M').apply(lambda r: r.start_time)
        df['event_period'] = df['event_date'].dt.to_period('M').apply(lambda r: r.start_time)
        period_label = 'месяц'

    df['retention_period'] = ((df['event_period'] - df['install_period']).dt.days //
                              (1 if period_type == 'День' else (7 if period_type == 'Неделя' else 30)))

    cohort = df.groupby(['install_period', 'retention_period'])['user_id'].nunique().reset_index()
    cohort_pivot = cohort.pivot(index='install_period', columns='retention_period', values='user_id')

    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0)

    st.subheader("\U0001F4CA Retention Table")
    st.dataframe(retention.fillna(0).style.format("{:.2%}"))

    st.subheader("\U0001F525 Retention Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(retention.fillna(0), annot=True, fmt=".0%", cmap="Blues", ax=ax)
    st.pyplot(fig)

    with st.expander("Скачать retention-таблицу"):
        csv = retention.fillna(0).to_csv(index=True).encode('utf-8')
        st.download_button("Скачать CSV", csv, file_name="retention_analysis.csv", mime='text/csv')

else:
    st.info("Ожидается загрузка CSV-файла.")
