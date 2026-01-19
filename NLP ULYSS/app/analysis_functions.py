import nltk
import spacy
import requests
import re
import time
import folium
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


# --- 1. Загрузка моделей и данных ---

def download_nltk_data():
    """Загружает все необходимые данные для NLTK."""
    packages = ['punkt', 'vader_lexicon', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    for package in packages:
        try:
            # Проверяем наличие пакета
            if package == 'punkt':
                nltk.data.find(f'tokenizers/{package}')
            elif package == 'vader_lexicon':
                nltk.data.find(f'sentiment/{package}')
            else:
                nltk.data.find(f'corpora/{package}')
        except LookupError:
            print(f"Загрузка пакета NLTK: {package}...")
            nltk.download(package, quiet=True)


def load_spacy_model(model_name="en_core_web_sm"):
    """Загружает модель spaCy, скачивая ее при необходимости."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Скачивание модели spaCy '{model_name}'...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    nlp.max_length = 2000000  # Увеличиваем лимит для обработки больших текстов
    return nlp


# --- 2. Основные функции анализа ---

def segment_text_by_character(text, character_names):
    """Сегментирует текст по персонажам, распознавая диалоги, начинающиеся с тире (—)."""
    segments = defaultdict(list)
    current_character = "Narrator"
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        attributed = False
        # Проверка на "Имя:"
        for char_name in character_names:
            if line.lower().startswith(char_name.lower() + ":"):
                current_character = char_name
                segments[current_character].append(line.split(":", 1)[1].strip())
                attributed = True
                break
        if attributed: continue

        # Проверка на диалог, начинающийся с тире
        if line.startswith('—'):
            segments["Unattributed Dialogue"].append(line[1:].strip())
        else:
            segments["Narrator"].append(line)

    return segments


def analyze_sentiment_over_time(text_segments):
    """Анализирует тональность для каждого сегмента текста."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {}
    for key, segments in text_segments.items():
        full_text = " ".join(segments)
        if full_text:
            sentiment_scores[key] = analyzer.polarity_scores(full_text)
    return sentiment_scores


def perform_thematic_modeling_bertopic(texts, min_topic_size=3):
    """Выполняет тематическое моделирование с помощью BERTopic."""
    if not texts or len(texts) < min_topic_size:
        return "Недостаточно текста для тематического моделирования."

    model = BERTopic(verbose=False, min_topic_size=min_topic_size)
    try:
        topics, _ = model.fit_transform(texts)
        return model.get_topic_info()
    except Exception as e:
        return f"Не удалось создать тематическую модель: {e}"


def analyze_language_style_spacy(text, nlp_model):
    """Анализирует стиль языка, используя spaCy для тегирования частей речи."""
    doc = nlp_model(text)
    pos_counts = defaultdict(int)
    for token in doc:
        pos_counts[token.pos_] += 1
    return {"pos_counts": dict(pos_counts)}


def extract_locations_ner(text, nlp_model):
    """Извлекает локации из текста, используя NER spaCy."""
    # Обрабатываем текст порциями, чтобы избежать ошибки с памятью
    doc = nlp_model(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]))


def geocode_locations(locations):
    """Преобразует названия локаций в координаты с помощью OpenStreetMap."""
    geocoded_locations = {}
    total = len(locations)
    for i, location in enumerate(locations):
        print(f"Геокодирование: {i + 1}/{total} - '{location}'...", end='\r')
        try:
            url = f"https://nominatim.openstreetmap.org/search?q={location}, Dublin&format=json&limit=1"
            response = requests.get(url, headers={'User-Agent': 'UlyssesNLP/1.0'}, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data:
                geocoded_locations[location] = (float(data[0]['lat']), float(data[0]['lon']))
        except requests.exceptions.RequestException:
            pass  # Игнорируем сетевые ошибки
        time.sleep(1)  # Соблюдаем политику API (1 запрос в секунду)
    print(f"\nГеокодирование завершено.")
    return geocoded_locations


# --- 3. Функции визуализации ---

def create_dublin_map(geocoded_locations):
    """Создает интерактивную карту Дублина с метками."""
    dublin_center = [53.3498, -6.2603]
    m = folium.Map(location=dublin_center, zoom_start=13, tiles='cartodbpositron')
    for location, (lat, lon) in geocoded_locations.items():
        folium.Marker(location=[lat, lon], popup=location).add_to(m)
    return m


def visualize_thought_transitions(segmented_text, sentiment_scores):
    """Визуализирует переходы мыслей между персонажами в виде графа."""
    G = nx.DiGraph()
    previous_key = None
    for key in segmented_text:
        sentiment = sentiment_scores.get(key, {}).get("compound", 0)
        G.add_node(key, sentiment=sentiment)
        if previous_key:
            G.add_edge(previous_key, key)
        previous_key = key

    pos = nx.spring_layout(G)
    node_colors = [G.nodes[node]['sentiment'] for node in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.RdYlGn, node_size=2500, font_size=10)
    plt.title("Граф переходов между персонажами")
    plt.show()

# --- 4. Функции для "музыкального" анализа (добавлены) ---

def identify_musical_phrases(text):
    """Анализирует ритмические характеристики текста, такие как длина предложений."""
    sentences = nltk.sent_tokenize(text)
    sentence_lengths = [len(word_tokenize(s)) for s in sentences if len(word_tokenize(s)) > 0]
    if not sentence_lengths:
        return {'sentence_lengths': [], 'avg_sentence_length': 0}
    avg_len = sum(sentence_lengths) / len(sentence_lengths)
    return {
        'sentence_lengths': sentence_lengths,
        'avg_sentence_length': avg_len
    }

def analyze_ngrams(text, n=2):
    """Анализирует N-граммы в тексте, исключая стоп-слова."""
    # Загружаем стоп-слова, если их еще нет
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))

    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stop_words]
    n_grams = ngrams(tokens, n)
    return Counter(n_grams)

def analyze_phonetic_patterns(text):
    """Анализирует фонетические паттерны, например, аллитерации."""
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    alliterative_sequences = []
    for i in range(len(tokens) - 1):
        # Простая проверка на аллитерацию: два слова подряд начинаются с одной буквы
        if tokens[i][0] == tokens[i+1][0]:
            alliterative_sequences.append(f"{tokens[i]} {tokens[i+1]}")
    return {'alliterative_sequences': alliterative_sequences}