# pages/3_AB_Test.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import beta, norm
import json
import os
from io import BytesIO
import yaml
import matplotlib.pyplot as plt
from utils.calc_helpers import pairwise_z_test

st.set_page_config(page_title="A/B Test Calculator", layout="wide")
st.title("A/B/n Test Calculator")

st.markdown("""
Введите данные по группам, чтобы рассчитать статистическую значимость A/B/n теста и сравнить варианты.
Также доступен Bayesian-анализ, экспорт отчёта и расчёт длительности эксперимента.
""")

st.subheader("Ввод данных")

load_exp = st.file_uploader("Загрузить конфигурацию эксперимента (YAML или JSON)", type=["yaml", "json"])

if load_exp:
    content = load_exp.read()
    if load_exp.name.endswith(".yaml"):
        loaded_data = yaml.safe_load(content)
    else:
        loaded_data = json.loads(content)
    groups = pd.DataFrame(loaded_data)
    st.success("Конфигурация успешно загружена")
else:
    st.markdown("Добавьте 2+ групп (название, кол-во пользователей, число конверсий)")
    groups = st.data_editor(
        pd.DataFrame({"Группа": ["A", "B"], "Пользователи": [1000, 1000], "Конверсии": [120, 150]}),
        num_rows="dynamic"
    )

if len(groups) < 2:
    st.warning("Нужно минимум две группы для анализа.")
else:
    groups["Конверсия"] = groups["Конверсии"] / groups["Пользователи"]
    st.subheader("Результаты по группам")
    st.dataframe(groups.style.format({"Конверсия": "{:.2%}"}))

    # Доверительные интервалы
    st.subheader("Доверительные интервалы")
    z = norm.ppf(0.975)
    groups["SE"] = np.sqrt(groups["Конверсия"] * (1 - groups["Конверсия"]) / groups["Пользователи"])
    groups["CI_low"] = groups["Конверсия"] - z * groups["SE"]
    groups["CI_high"] = groups["Конверсия"] + z * groups["SE"]

    fig_ci, ax_ci = plt.subplots(figsize=(8, 5))
    ax_ci.errorbar(groups["Группа"], groups["Конверсия"],
                   yerr=z * groups["SE"], fmt='o', capsize=5)
    ax_ci.set_title("Доверительные интервалы конверсий")
    ax_ci.set_ylabel("Конверсия")
    st.pyplot(fig_ci)

    # Расчёт длительности эксперимента
    st.subheader("⏱ Расчёт длительности эксперимента")
    baseline_rate = st.number_input("Базовая конверсия (%)", value=10.0, step=0.1) / 100
    mde = st.number_input("Минимальный детектируемый эффект (MDE, %)", value=10.0, step=0.1) / 100
    power = st.slider("Желаемая мощность теста (power)", 0.7, 0.99, 0.8)
    alpha = 0.05
    pooled_p = baseline_rate + mde
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    pooled_std = np.sqrt(baseline_rate * (1 - baseline_rate) + pooled_p * (1 - pooled_p))
    sample_size = 2 * ((z_alpha + z_beta) * pooled_std / mde) ** 2
    st.markdown(f"**Рекомендуемый размер выборки на группу:** {int(sample_size):,} пользователей")

    # ROMI-график
    st.subheader("Сравнение ROMI по группам")
    revenue_per_user = st.number_input("Доход на пользователя ($)", value=20.0, step=1.0)
    cost_per_user = st.number_input("Стоимость привлечения ($)", value=10.0, step=1.0)
    groups["ROMI"] = (groups["Конверсия"] * revenue_per_user - cost_per_user) / cost_per_user
    fig_romi, ax_romi = plt.subplots(figsize=(8, 4))
    ax_romi.bar(groups["Группа"], groups["ROMI"])
    ax_romi.set_title("ROMI по группам")
    ax_romi.set_ylabel("ROMI")
    st.pyplot(fig_romi)

    # Попарные сравнения
    st.subheader("Попарное сравнение (Z-тест)")
    results = pairwise_z_test(groups)
    st.dataframe(pd.DataFrame(results).style.format({"Разница": "{:.2%}", "p-value": "{:.4f}"}))

    # Bayesian
    st.subheader("Bayesian A/B")
    st.markdown("Сравнение через распределения Beta (априори = 1, 1)")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 1, 500)
    for idx, row in groups.iterrows():
        a, b = row["Конверсии"] + 1, row["Пользователи"] - row["Конверсии"] + 1
        ax.plot(x, beta.pdf(x, a, b), label=f"{row['Группа']}")
    ax.set_title("Beta distributions")
    ax.legend()
    st.pyplot(fig)

    # Экспорт YAML конфигурации
    st.subheader("Сохранение эксперимента в YAML")
    exp_name = st.text_input("Название эксперимента", "experiment_1")
    if st.button("Сохранить как YAML"):
        os.makedirs("data/experiments", exist_ok=True)
        with open(f"data/experiments/{exp_name}.yaml", "w") as f:
            yaml.dump(groups.to_dict(orient="list"), f, allow_unicode=True)
        st.success(f"Сохранено в data/experiments/{exp_name}.yaml")

    # Экспорт Excel
    st.subheader("Экспорт результатов")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        groups.to_excel(writer, sheet_name="Groups", index=False)
        pd.DataFrame(results).to_excel(writer, sheet_name="Pairwise Test", index=False)
    st.download_button("⬇Скачать Excel отчет", buffer.getvalue(), file_name="abtest_report.xlsx")
