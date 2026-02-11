#Ulysses NLP Analysis Project

A comprehensive NLP analysis of James Joyce's novel Ulysses, incorporating multiple analytical approaches and an interactive educational app.

## Project Overview

This project is a multifaceted NLP analysis of James Joyce's modernist masterpiece Ulysses, examining the novel through seven different analytical lenses. The project is divided into two main parts:

1. Research Notebook (Jupyter Notebook): A detailed technical walkthrough demonstrating each analysis method step by step. Ideal for researchers and students studying NLP.
2. Interactive Web App (Streamlit): A convenient tool for exploring excerpts from Ulysses using real-time NLP analysis.

### Analytical Approaches

1. **Stream of Consciousness Analysis** - Character Segmentation and Emotion Tracking
2. **Allusion and Reference Detection** - Intertextual Analysis and Literary Entity Recognition
3. **Joycean Style Imitation** - Generative Modeling of Joyce's Unique Writing Style
4. **Dublin Voices** - Character Speech Classification and Linguistic Profiling
5. **Geo-Literary Reconstruction** - Mapping the Geography of "One Day in Dublin"
6. **Language as Music** - Analysis of Rhythmic and Phonetic Patterns
7. **Educational App** - Interactive Web Tool for Novel Research

## Project Structure

```
ulysses_nlp_project/
├── app/
│ ├── analysis_functions.py # Module with analysis functions
│ └── ulysses_guide_app.py # Streamlit application
├── notebook/
│ └── ulysses_nlp_research.ipynb # Jupyter research notebook
├── data/
│ ├── chapter_01.txt # Sample text
│ └── chapter_02.txt # Sample text
├── requirements.txt # Python dependencies
└── README.md # This file
```

## Installation

### Prerequisites

- Python 3.11+
- Jupyter Notebook or JupyterLab
- Virtual Environment (Recommended)

### Setup

1. Clone or extract the project files.
2. Create and activate the virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Download the required NLTK data and spaCy models:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
python -m spacy download en_core_web_sm
```

## Usage

### 1. Research Notebook

For a deep dive into analysis methods, open and launch Jupyter Notebook.

```bash
cd notebook/
jupyter notebook ulysses_nlp_research.ipynb
# or jupyter lab ulysses_nlp_research.ipynb
```
The notebook will guide you through all the analysis steps with code examples and visualizations.

### 2. Interactive Web Application

For interactive text analysis, launch the Streamlit application.

```bash
cd app/
streamlit run ulysses_guide_app.py
```
The application will be available at `http://localhost:8501`.

