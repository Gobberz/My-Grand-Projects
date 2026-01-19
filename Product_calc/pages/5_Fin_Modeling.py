# pages/5_Fin_Modeling.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Product Financial Model", layout="wide")
st.title("Финансовая модель продукта")

st.markdown("""
Прогнозируйте выручку, LTV, прибыль и оценивайте сценарии Best/Base/Worst:
- What-if симуляции
- Финансовый план по сегментам
""")

st.subheader("Ввод данных по сегментам")
data = st.data_editor(
    pd.DataFrame({
        "Сегмент": ["Free", "Premium", "Enterprise"],
        "Пользователи": [10000, 1500, 100],
        "ARPU": [0.5, 15.0, 50.0],
        "Retention": [0.3, 0.5, 0.8],
        "Margin": [0.5, 0.7, 0.8]
    }),
    num_rows="dynamic"
)

st.subheader("What-if симуляция")
conv_change = st.slider("Изменение ARPU (%)", -50, 50, 0)
ret_change = st.slider("Изменение Retention (%)", -50, 50, 0)

data["ARPU_adj"] = data["ARPU"] * (1 + conv_change / 100)
data["Retention_adj"] = data["Retention"] * (1 + ret_change / 100)
data["LTV"] = data["ARPU_adj"] * data["Retention_adj"] * data["Margin"]
data["Revenue"] = data["Пользователи"] * data["ARPU_adj"]
data["Total_LTV"] = data["Пользователи"] * data["LTV"]

st.subheader("Финансовый прогноз")
display_df = data[["Сегмент", "Пользователи", "ARPU_adj", "Retention_adj", "LTV", "Revenue", "Total_LTV"]] \
    .rename(columns={"ARPU_adj": "ARPU", "Retention_adj": "Retention"})

st.dataframe(display_df.style.format({
    "Пользователи": "{:,.0f}",
    "ARPU": "{:.2f}",
    "Retention": "{:.2f}",
    "LTV": "{:.2f}",
    "Revenue": "{:.2f}",
    "Total_LTV": "{:.2f}"
}))


fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(data["Сегмент"], data["Revenue"], label="Revenue")
ax.bar(data["Сегмент"], data["Total_LTV"], bottom=data["Revenue"], label="Total LTV", alpha=0.5)
ax.set_ylabel("$")
ax.set_title("Выручка и пожизненная ценность")
ax.legend()
st.pyplot(fig)

st.subheader("Сценарный анализ")
scenarios = pd.DataFrame({
    "Сценарий": ["Best", "Base", "Worst"],
    "ARPU": [20, 15, 10],
    "Retention": [0.8, 0.5, 0.2],
    "Margin": [0.8, 0.7, 0.5]
})
scenarios["LTV"] = scenarios["ARPU"] * scenarios["Retention"] * scenarios["Margin"]
scenarios["Revenue"] = data["Пользователи"].sum() * scenarios["ARPU"]
scenarios["Total_LTV"] = data["Пользователи"].sum() * scenarios["LTV"]

st.dataframe(scenarios.style.format({
    "ARPU": "{:.2f}",
    "Retention": "{:.2f}",
    "Margin": "{:.2f}",
    "LTV": "{:.2f}",
    "Revenue": "{:,.0f}",
    "Total_LTV": "{:,.0f}"
}))

