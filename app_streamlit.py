# app_streamlit.py
import streamlit as st
from fetcher import fetch_article_from_url, fetch_articles_from_rss
from preprocess import clean_text, extract_entities, split_text_for_model
from summarizer import build_summarizer, summarize_long_text, summarize_text
from transformers import logging
logging.set_verbosity_error()

st.set_page_config(page_title="NLP News Summarizer", layout="centered")

@st.cache_resource
def load_model(model_name="facebook/bart-large-cnn"):
    return build_summarizer(model_name)

st.title("📰 NLP News Summarizer — spaCy + Hugging Face")
st.markdown("Paste a news article URL or raw text. The app extracts content, shows named entities, and produces an abstractive summary.")

mode = st.radio("Input mode", ["URL", "Raw text", "RSS feed"])

model_name = st.selectbox("Model (Hugging Face)", ["facebook/bart-large-cnn", "sshleifer/distilbart-cnn-12-6"])
summarizer = load_model(model_name)

if mode == "URL":
    url = st.text_input("Article URL")
    if st.button("Fetch & Summarize") and url.strip():
        with st.spinner("Fetching article..."):
            try:
                article = fetch_article_from_url(url)
            except Exception as e:
                st.error(f"Failed to fetch article: {e}")
                st.stop()
        text = clean_text(article["text"])
        st.subheader(article.get("title", ""))
        st.caption(f"Authors: {article.get('authors','')} • Published: {article.get('publish_date','')}")
        if not text:
            st.warning("No article text found.")
            st.stop()
        with st.spinner("Extracting entities and running summarizer..."):
            ents = extract_entities(text)
            chunks = split_text_for_model(text, max_tokens_estimate=900)
            summary = summarize_long_text(summarizer, chunks, max_length=160, min_length=40)
        st.subheader("Summary")
        st.write(summary)
        st.subheader("Named Entities")
        if ents:
            st.json(ents)
        else:
            st.write("No named entities found.")
        st.subheader("Article Text (first 1000 chars)")
        st.write(text[:1000] + ("..." if len(text)>1000 else ""))

elif mode == "Raw text":
    txt = st.text_area("Paste article text here", height=300)
    max_len = st.slider("Summary max length (words)", 50, 400, 150)
    if st.button("Summarize text") and txt.strip():
        text = clean_text(txt)
        with st.spinner("Processing..."):
            ents = extract_entities(text)
            chunks = split_text_for_model(text, max_tokens_estimate=900)
            summary = summarize_long_text(summarizer, chunks, max_length=max_len, min_length=30)
        st.subheader("Summary")
        st.write(summary)
        st.subheader("Entities")
        st.json(ents)

else:  # RSS
    rss = st.text_input("RSS feed URL (e.g., https://rss.cnn.com/rss/edition.rss)")
    limit = st.number_input("Max articles to fetch", min_value=1, max_value=10, value=3)
    if st.button("Fetch RSS"):
        with st.spinner("Fetching feed..."):
            items = fetch_articles_from_rss(rss, limit=limit)
        if not items:
            st.warning("No items fetched; check RSS URL.")
        for i, it in enumerate(items, 1):
            st.markdown(f"### {i}. {it.get('title','(no title)')}")
            text = clean_text(it.get("text",""))
            chunks = split_text_for_model(text, max_tokens_estimate=800)
            summary = summarize_long_text(summarizer, chunks, max_length=120, min_length=30)
            st.write(summary)
            st.write("---")

st.markdown("----\nProject: spaCy preprocessing + Hugging Face abstractive summarization. Adjust model choices and summary lengths for different behavior.")
