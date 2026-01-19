import streamlit as st
import pandas as pd
from ab_test_calculator import ABTestCalculator

st.set_page_config(page_title="A/B Test Calculator", layout="centered")
st.title("📊 A/B Test Calculator")

st.markdown("Upload your CSV or manually input data for A/B testing.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    alpha = st.slider("Significance level (alpha)", 0.001, 0.1, 0.05, 0.001)
    delta = st.slider("Minimum effect (delta)", 0.0, 0.1, 0.0, 0.001)
    method = st.selectbox("Statistical Method", ["z_test", "bootstrap", "bayesian"])
    alternative = st.selectbox("Alternative Hypothesis", ["two-sided", "greater", "less"])

calc = ABTestCalculator(alpha=alpha, delta=delta, method=method, alternative=alternative)

# Input block
option = st.radio("Input Method", ["Manual", "Upload CSV"])

if option == "Manual":
    col1, col2 = st.columns(2)
    with col1:
        n_A = st.number_input("Users in Group A", min_value=1, value=1000)
        conv_A = st.number_input("Conversions in Group A", min_value=0, value=120)
    with col2:
        n_B = st.number_input("Users in Group B", min_value=1, value=980)
        conv_B = st.number_input("Conversions in Group B", min_value=0, value=138)

    if st.button("Run Test"):
        results = calc.analyze(n_A, conv_A, n_B, conv_B)
        st.success("Test completed.")
        st.code(calc.summarize(), language="markdown")
        if method == "bootstrap":
            calc.plot_bootstrap()
            st.pyplot()

else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Preview", df.head())

        try:
            calc.from_dataframe(df)
            st.success("Test completed from CSV.")
            st.code(calc.summarize(), language="markdown")
            if method == "bootstrap":
                calc.plot_bootstrap()
                st.pyplot()
        except Exception as e:
            st.error(f"Error analyzing file: {e}")
