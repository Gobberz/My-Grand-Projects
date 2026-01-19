# app.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Product Analytics Calculator", layout="wide")

# Навигация автоматически работает, если есть файлы в /pages
st.sidebar.title("Меню")
st.sidebar.info("Выберите вкладку:")

st.title("Product Analytics Calculator")

st.markdown("""
Добро пожаловать! Используйте меню слева для перехода между вкладками:

**Доступные модули:**
- Retention Analysis
- LTV / CAC Calculator
- A/B Test Calculator

Для работы загрузите ваши данные в нужной вкладке.
""")
