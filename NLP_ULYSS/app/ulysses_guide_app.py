import streamlit as st
import pandas as pd
from analysis_functions import * # Импортируем ВСЕ из нашей чистой библиотеки

# Загружаем модели один раз при старте приложения, чтобы не делать это при каждом клике
@st.cache_resource
def load_models():
    print("Загрузка моделей для приложения...")
    download_nltk_data()
    nlp_model = load_spacy_model()
    print("Модели загружены.")
    return nlp_model

nlp = load_models()

st.set_page_config(page_title="Ulysses Guide", layout="wide")
st.title("📚 Ulysses NLP Guide")
st.markdown("*Интерактивное приложение для исследования «Улисса» с помощью NLP*")

analysis_type = st.sidebar.selectbox(
    "Выберите тип анализа:",
    ["Анализ потока сознания", "Гео-литературное картирование"]
)

user_text = st.text_area(
    "Введите отрывок из «Улисса»:",
    height=250,
    value="""Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.
—Introibo ad altare Dei.
Halted, he peered down the dark winding stairs and called out coarsely:
—Come up, Kinch! Come up, you fearful jesuit!"""
)

if st.button("🔍 Анализировать", type="primary"):
    if user_text.strip():
        with st.spinner("Анализ..."):
            if analysis_type == "Анализ потока сознания":
                st.subheader("🧠 Анализ потока сознания")
                segments = segment_text_by_character(user_text, ["Stephen", "Buck Mulligan"])
                sentiment = analyze_sentiment_over_time(segments)
                st.dataframe(pd.DataFrame(sentiment).T.reset_index().rename(columns={'index': 'Персонаж'}))

            elif analysis_type == "Гео-литературное картирование":
                st.subheader("🗺️ Гео-литературная реконструкция")
                locations = extract_locations_ner(user_text, nlp)
                if locations:
                    st.write("**Найденные локации:**", ", ".join(locations))
                    # В приложении геокодирование может быть долгим, поэтому пока просто выводим
                    st.info("В Jupyter-ноутбуке эти локации отображаются на интерактивной карте.")
                else:
                    st.write("Локации не найдены в этом фрагменте.")
    else:
        st.error("Пожалуйста, введите текст для анализа.")