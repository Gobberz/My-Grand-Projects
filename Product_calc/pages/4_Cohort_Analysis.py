# pages/4_Cohort_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cohort Analysis", layout="wide")
st.title("Когортный анализ пользователей")

st.markdown("""
Загрузите CSV с полями `user_id`, `install_date`, `event_date`, `revenue`, чтобы провести когортный анализ:
- Retention по когортам
- Доход/пользователь
- LTV когорт
""")

file = st.file_uploader("Загрузите CSV", type=["csv"])

if file:
    df = pd.read_csv(file, parse_dates=["install_date", "event_date"])
    df["install_month"] = df["install_date"].dt.to_period("M").dt.to_timestamp()
    df["event_month"] = df["event_date"].dt.to_period("M").dt.to_timestamp()
    df["cohort_period"] = ((df["event_month"] - df["install_month"]) / np.timedelta64(1, 'M')).round().astype("int")

    cohort_data = df.groupby(["install_month", "cohort_period"])\
        .agg(users=("user_id", "nunique"), revenue=("revenue", "sum")).reset_index()

    cohort_pivot = cohort_data.pivot(index="install_month", columns="cohort_period", values="users")
    base_users = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(base_users, axis=0)

    st.subheader("Retention (по месяцам)")
    st.dataframe(retention.fillna(0).style.format("{:.2%}"))

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.heatmap(retention.fillna(0), annot=True, fmt=".0%", cmap="YlGnBu", ax=ax1)
    st.pyplot(fig1)

    revenue_pivot = cohort_data.pivot(index="install_month", columns="cohort_period", values="revenue")
    revenue_per_user = revenue_pivot.divide(cohort_pivot)
    ltv = revenue_per_user.cumsum(axis=1)

    st.subheader("LTV когорт")
    st.dataframe(ltv.fillna(0).style.format("{:.2f}"))

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(ltv.fillna(0), annot=True, fmt=".1f", cmap="Oranges", ax=ax2)
    st.pyplot(fig2)
else:
    st.info("Загрузите CSV-файл с нужными полями для анализа.")
