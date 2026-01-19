# pages/2_LTV_CAC.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="LTV & CAC Calculator", layout="wide")
st.title("LTV / CAC Calculator")

st.markdown("""
Введите данные вручную или загрузите файл со столбцами:
- `segment`, `ARPU`, `Retention`, `Margin`, `CAC`
""")

uploaded_file = st.file_uploader("Загрузите CSV (необязательно)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.subheader("Ввод вручную")
    with st.form("manual_input"):
        segment = st.text_input("Segment", "Default")
        ARPU = st.number_input("ARPU ($)", 0.0, 1000.0, 50.0)
        Retention = st.slider("Retention (0-1)", 0.0, 1.0, 0.4)
        Margin = st.slider("Margin (0-1)", 0.0, 1.0, 0.7)
        CAC = st.number_input("CAC ($)", 0.0, 1000.0, 30.0)
        submitted = st.form_submit_button("Рассчитать")

    if submitted:
        df = pd.DataFrame([{
            "segment": segment,
            "ARPU": ARPU,
            "Retention": Retention,
            "Margin": Margin,
            "CAC": CAC
        }])

if 'df' in locals():
    df['LTV'] = df['ARPU'] * df['Retention'] * df['Margin']
    df['ROMI'] = df['LTV'] / df['CAC']
    df['Payback_Period'] = df['CAC'] / (df['ARPU'] * df['Margin'])
    df['Profit_per_User'] = df['LTV'] - df['CAC']

    st.subheader("Результаты расчета")
    st.dataframe(df.style.format("{:.2f}"))

    st.subheader("Сравнение LTV и CAC")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df['segment'], df['LTV'], label='LTV')
    ax.bar(df['segment'], df['CAC'], label='CAC', alpha=0.7)
    ax.set_ylabel("$")
    ax.legend()
    st.pyplot(fig)

    with st.expander("Скачать результаты"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Скачать CSV", csv, file_name="ltv_cac_results.csv", mime='text/csv')
else:
    st.info("Ожидается ввод данных или загрузка файла.")
