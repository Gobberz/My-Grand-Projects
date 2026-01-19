import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Прогноз по регионам и категориям", layout="wide")
st.title("Калькулятор прогнозов по регионам и категориям")

# Исходные данные
st.sidebar.header("Исходные параметры")
price = st.sidebar.number_input("Базовая цена (₽)", 0.0, value=2000.0)
cost = st.sidebar.number_input("Себестоимость (₽)", 0.0, value=1200.0)
plan_sales = st.sidebar.number_input("План продаж (шт/мес)", 0, value=1000)
marketing_budget = st.sidebar.number_input("Маркетинг (₽)", 0.0, value=50000.0)
marketing_impact = st.sidebar.number_input("Маркетинг-эффект (₽)", 0.0, value=70000.0)
fixed_costs = st.sidebar.number_input("Фиксированные издержки (₽)", 0.0, value=100000.0)
variable_costs = st.sidebar.number_input("Переменные издержки (₽)", 0.0, value=400000.0)
tax_rate = st.sidebar.slider("Налоговая ставка (%)", 0.0, 50.0, 20.0)
n_outlets = st.sidebar.number_input("Число торговых точек", 1, value=5)

# Эластичность
with st.expander("Параметры эластичности и конкуренции"):
    price_elasticity = st.slider("Ценовая эластичность", -5.0, 0.0, -1.5)
    ad_elasticity = st.slider("Маркетинговая эластичность", 0.0, 3.0, 0.5)
    competitor_influence = st.slider("Конкурентное давление", 0.1, 2.0, 1.0)

# Прогноз 
st.sidebar.header("Настройки прогноза")
n_months = st.sidebar.slider("Горизонт прогноза (мес)", 1, 12, 6)
with st.expander("Темпы роста"):
    monthly_sales_growth = st.slider("Рост плана продаж (%)", -50, 100, 5)
    monthly_price_growth = st.slider("Рост цены (%)", -20, 20, 0)
    monthly_cost_growth = st.slider("Рост себестоимости (%)", 0, 20, 2)
    monthly_marketing_growth = st.slider("Рост маркетинга (%)", 0, 50, 5)

scale_effect = st.checkbox("Учитывать эффект масштабирования", value=True)

# Загрузка и редактирование параметров регионов/категорий ===
st.subheader("Параметры по регионам и категориям")
example_data = pd.DataFrame({
    "Регион": ["Город_1", "Город_2", "Город_3"],
    "Категория": ["Товар_группа_1", "Товар_группа_2", "Товар_группа_3"],
    "Коэф. спроса": [1.0, 0.8, 0.9],
    "Локальная наценка (%)": [10, 5, 7],
    "Издержки (%)": [5, 3, 4]
})

df_input = st.data_editor(example_data, num_rows="dynamic", use_container_width=True)

# Расчёты
def calculate_extended(row, scale_effect=True):
    try:
        if scale_effect:
            scale_factor = min(1.0, 1000 / max(row.fact_sales, 1))
            variable_costs_scaled = row.variable_costs * scale_factor
            variable_cost_per_unit = variable_costs_scaled / row.fact_sales if row.fact_sales > 0 else 0
        else:
            variable_costs_scaled = row.variable_costs
            variable_cost_per_unit = variable_costs_scaled / row.fact_sales if row.fact_sales > 0 else 0

        def scaled_fixed_costs(base_fixed_costs, outlets):
            steps = outlets // 10
            return base_fixed_costs * (1 + steps * 0.15)

        fixed_costs_scaled = scaled_fixed_costs(row.fixed_costs, row.n_outlets)

        revenue = row.fact_sales * row.price
        gross_profit = revenue - (row.fact_sales * row.cost) - variable_costs_scaled
        taxable_base = gross_profit - fixed_costs_scaled
        taxes = max(0, taxable_base * (row.tax_rate / 100))
        net_profit = gross_profit - fixed_costs_scaled - taxes

        romi = (row.marketing_impact / row.marketing_budget) * 100 if row.marketing_budget > 0 else 0
        roi_region = (net_profit / (row.marketing_budget + fixed_costs_scaled + variable_costs_scaled)) * 100

        return pd.Series([revenue, net_profit, romi, roi_region])
    except Exception:
        return pd.Series([None] * 4)

def forecast_scenario(row, scenario_name, n_months, sales_growth, price_growth, cost_growth, marketing_growth,
                      base_price, base_marketing_budget, base_plan_sales,
                      price_elasticity, ad_elasticity, competitor_influence,
                      scale_effect=True):
    months, revenue_list, profit_list, romi_list, roi_list = [], [], [], [], []
    for month in range(1, n_months + 1):
        row = row.copy()
        row.price *= (1 + price_growth / 100) ** month
        row.cost *= (1 + cost_growth / 100) ** month
        row.marketing_budget *= (1 + marketing_growth / 100) ** month
        row.plan_sales *= (1 + sales_growth / 100) ** month

        price_change = (row.price - base_price) / base_price if base_price else 0
        ad_change = (row.marketing_budget - base_marketing_budget) / base_marketing_budget if base_marketing_budget else 0

        delta_sales_price = price_elasticity * price_change
        delta_sales_ad = ad_elasticity * ad_change
        competition_effect = 1 / competitor_influence
        sales_multiplier = (1 + delta_sales_price + delta_sales_ad) * competition_effect
        row.fact_sales = max(0, row.plan_sales * sales_multiplier)

        metrics = calculate_extended(row, scale_effect=scale_effect)
        months.append(month)
        revenue_list.append(metrics[0])
        profit_list.append(metrics[1])
        romi_list.append(metrics[2])
        roi_list.append(metrics[3])

    return pd.DataFrame({
        "Месяц": months,
        "Сценарий": scenario_name,
        "Выручка": revenue_list,
        "Чистая прибыль": profit_list,
        "ROMI": romi_list,
        "ROI региона": roi_list
    })

def apply_scenario(row, scenario: str):
    row = row.copy()
    if scenario == "Оптимистичный":
        row.price *= 1.05
        row.marketing_budget *= 1.3
    elif scenario == "Пессимистичный":
        row.price *= 0.95
        row.marketing_budget *= 0.7
    return row

#Прогноз по всем строкам
forecast_all = []

for _, row_cfg in df_input.iterrows():
    region = row_cfg["Регион"]
    category = row_cfg["Категория"]
    demand_coeff = row_cfg["Коэф. спроса"]
    markup = row_cfg["Локальная наценка (%)"] / 100
    extra_cost = row_cfg["Издержки (%)"] / 100

    base_price_loc = price * (1 + markup)
    cost_loc = cost * (1 + extra_cost)

    row_local = pd.Series({
        'product_name': category,
        'price': base_price_loc,
        'cost': cost_loc,
        'plan_sales': plan_sales * demand_coeff,
        'marketing_budget': marketing_budget,
        'marketing_impact': marketing_impact,
        'fixed_costs': fixed_costs,
        'variable_costs': variable_costs,
        'tax_rate': tax_rate,
        'n_outlets': n_outlets,
        'fact_sales': plan_sales
    })

    for scenario in ["Базовый", "Оптимистичный", "Пессимистичный"]:
        scenario_row = apply_scenario(row_local.copy(), scenario)
        df_forecast = forecast_scenario(
            scenario_row, scenario, n_months,
            monthly_sales_growth, monthly_price_growth, monthly_cost_growth, monthly_marketing_growth,
            price, marketing_budget, plan_sales,
            price_elasticity, ad_elasticity, competitor_influence,
            scale_effect=scale_effect
        )
        df_forecast["Регион"] = region
        df_forecast["Категория"] = category
        forecast_all.append(df_forecast)

forecast_total = pd.concat(forecast_all)

# График
st.subheader("Динамика чистой прибыли по категориям и регионам")
fig = px.line(
    forecast_total,
    x="Месяц",
    y="Чистая прибыль",
    color="Сценарий",
    line_dash="Регион",
    facet_col="Категория",
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Общий прогноз по всем данным")
st.dataframe(forecast_total, use_container_width=True)