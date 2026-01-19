# pages/6_Marketing_Analytics.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Marketing Analytics", layout="wide")
st.title("Маркетинговая аналитика")

st.markdown("""
Этот модуль включает:
- Анализ воронки маркетинга
- Расчёт CPA, CTR, ROI, ROMI
- What-if симулятор
- Канальный анализ
""")

# Ввод данных воронки
st.subheader("Маркетинговая воронка")
views = st.number_input("Показы", min_value=1, value=10000)
clicks = st.number_input("Клики", min_value=0, value=1500)
leads = st.number_input("Лиды", min_value=0, value=300)
clients = st.number_input("Клиенты", min_value=0, value=60)
cost = st.number_input("Общий бюджет кампании ($)", min_value=0.0, value=1500.0)

ctr = clicks / views
cr = clients / clicks if clicks > 0 else 0
cpl = cost / leads if leads > 0 else 0
cpa = cost / clients if clients > 0 else 0
roi = (clients * 100 - cost) / cost if cost > 0 else 0

st.metric("CTR", f"{ctr:.2%}")
st.metric("CR (клиенты/клики)", f"{cr:.2%}")
st.metric("CPL ($)", f"{cpl:.2f}")
st.metric("CPA ($)", f"{cpa:.2f}")
st.metric("ROI", f"{roi:.2%}")

# What-if симулятор
st.subheader("What-if анализ")
ctr_delta = st.slider("Изменение CTR (%)", -50, 50, 0)
cr_delta = st.slider("Изменение CR (%)", -50, 50, 0)
cost_delta = st.slider("Изменение бюджета (%)", -50, 50, 0)

sim_views = views
sim_clicks = views * (ctr * (1 + ctr_delta / 100))
sim_clients = sim_clicks * (cr * (1 + cr_delta / 100))
sim_cost = cost * (1 + cost_delta / 100)
sim_cpa = sim_cost / sim_clients if sim_clients > 0 else 0
sim_roi = (sim_clients * 100 - sim_cost) / sim_cost if sim_cost > 0 else 0

st.markdown(f"**Прогнозируемые клики:** {sim_clicks:,.0f}")
st.markdown(f"**Прогнозируемые клиенты:** {sim_clients:,.0f}")
st.markdown(f"**Прогнозируемый CPA:** ${sim_cpa:,.2f}")
st.markdown(f"**Прогнозируемый ROI:** {sim_roi:.2%}")

# Канальный анализ
st.subheader("Канальный анализ")
channels = st.data_editor(pd.DataFrame({
    "Канал": ["Facebook", "Google", "Email", "SEO"],
    "Бюджет": [500, 600, 200, 200],
    "Клики": [800, 1000, 300, 200],
    "Клиенты": [30, 40, 10, 5]
}), num_rows="dynamic")

channels["CTR"] = channels["Клики"] / views
channels["CPA"] = channels["Бюджет"] / channels["Клиенты"]
channels["ROAS"] = (channels["Клиенты"] * 100) / channels["Бюджет"]

st.dataframe(channels.style.format({
    "Бюджет": "${:,.0f}",
    "CTR": "{:.2%}",
    "CPA": "${:,.2f}",
    "ROAS": "{:.2f}x"
}))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(channels["Канал"], channels["ROAS"], color="skyblue")
ax.set_title("ROAS по каналам")
ax.set_ylabel("ROAS (x)")
ax.axhline(1, color='red', linestyle='--')
st.pyplot(fig)
