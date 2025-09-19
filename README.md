NLP News Summarizer (spaCy + Hugging Face Transformers)

Quick start:
1. python -m venv venv && source venv/bin/activate (or venv\Scripts\activate on Windows)
2. pip install -r requirements.txt
3. python -m spacy download en_core_web_sm
4. streamlit run app_streamlit.py

Features:
- Fetch article from URL (newspaper3k)
- Preprocess: sentence/phrase cleaning + spaCy NER
- Abstractive summarization using a Hugging Face model (facebook/bart-large-cnn by default)
- Streamlit UI to paste URL or raw text
