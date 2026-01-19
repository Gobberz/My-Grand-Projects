# pages/7_Unit_Economics.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Unit Economics", layout="wide")
st.title("Юнит-экономика")

st.markdown("""
Анализируйте экономику продукта по ключевым метрикам:
- ARPU / ARPPU, CAC, LTV, GPM, Retention Cost, Contribution Margin
- Payback и LTV/CAC
- Моделирование по сегментам и сравнительный анализ
""")

st.subheader("Ввод данных")
segments = st.number_input("Количество сегментов (например, продуктов или каналов)", min_value=1, max_value=10, value=2)

segment_data = []

for i in range(segments):
    st.markdown(f"### Сегмент {i+1}")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input(f"Название сегмента {i+1}", value=f"Сегмент {i+1}", key=f"name_{i}")
        users = st.number_input(f"Всего пользователей - {name}", min_value=1, value=10000, key=f"users_{i}")
        paying = st.number_input(f"Платящих пользователей - {name}", min_value=1, value=2000, key=f"paying_{i}")
        revenue = st.number_input(f"Выручка ($) - {name}", min_value=0.0, value=30000.0, key=f"revenue_{i}")
        marketing = st.number_input(f"Маркетинг ($) - {name}", min_value=0.0, value=10000.0, key=f"marketing_{i}")
    with col2:
        var_cost = st.number_input(f"Переменные затраты на пользователя - {name}", min_value=0.0, value=1.0, key=f"vcost_{i}")
        retention = st.slider(f"Retention (мес) - {name}", 1, 36, 6, key=f"retention_{i}")
        gpm = st.slider(f"Валовая маржа (%) - {name}", 0, 100, 70, key=f"gpm_{i}")

    arpu = revenue / users
    arppu = revenue / paying
    cac = marketing / paying if paying > 0 else 0
    ltv = arpu * retention * (gpm / 100)
    ltv_cac = ltv / cac if cac > 0 else 0
    payback = cac / (arpu * (gpm / 100)) if arpu > 0 else 0
    contribution = arpu - var_cost
    retention_cost = var_cost * retention

    segment_data.append({
        "Сегмент": name,
        "ARPU": arpu,
        "ARPPU": arppu,
        "CAC": cac,
        "LTV": ltv,
        "LTV/CAC": ltv_cac,
        "Payback": payback,
        "Contribution Margin": contribution,
        "Retention Cost": retention_cost,
        "GPM (%)": gpm
    })

# Таблица результатов
df = pd.DataFrame(segment_data)
st.subheader("Сравнительная таблица по сегментам")
st.dataframe(df.style.format("{:.2f}"))

# Визуализация сравнения LTV/CAC
st.subheader("Сравнение LTV/CAC по сегментам")
if not df.empty:
    fig_ltv_cac, ax_ltv_cac = plt.subplots(figsize=(10, 4))
    ax_ltv_cac.bar(df["Сегмент"], df["LTV/CAC"], color="skyblue")
    ax_ltv_cac.set_ylabel("LTV/CAC")
    ax_ltv_cac.set_title("Сравнение LTV/CAC по сегментам")
    st.pyplot(fig_ltv_cac)

# График окупаемости для первого сегмента
st.subheader("Окупаемость по первому сегменту")
if segment_data:
    first = segment_data[0]
    months = list(range(1, int(retention)+1))
    cumulative = [first['ARPU'] * m * (first['GPM (%)'] / 100) for m in months]
    cac_line = [first['CAC']] * len(months)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(months, cumulative, label="Кумулятивный доход", linewidth=2)
    ax.plot(months, cac_line, '--', label="CAC", color='red')
    ax.set_xlabel("Месяц")
    ax.set_ylabel("$")
    ax.set_title(f"Окупаемость: {first['Сегмент']}")
    ax.legend()
    st.pyplot(fig)

# Экспорт в Excel
st.subheader("Экспорт в Excel")
excel_buffer = io.BytesIO()
df.to_excel(excel_buffer, index=False, engine='openpyxl')
st.download_button(
    label="Скачать таблицу Excel",
    data=excel_buffer.getvalue(),
    file_name="unit_economics.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
