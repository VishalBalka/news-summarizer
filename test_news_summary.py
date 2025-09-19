# file: test_news_summary.py
from newspaper import Article
from transformers import pipeline

# 👇 THIS is the only line you replace with your own news article link
url = "https://www.bbc.com/news/articles/c0q7l8ewj0wo"


try:
    article = Article(url)   # fetches article from the URL
    article.download()
    article.parse()

    print("\n--- ARTICLE TITLE ---")
    print(article.title)

    print("\n--- ORIGINAL TEXT (first 500 chars) ---")
    print(article.text[:500])

    # Load Hugging Face summarizer
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Summarize first 1000 characters (to avoid model overload)
    summary = summarizer(article.text[:1000], max_length=150, min_length=50, do_sample=False)
    print("\n--- SUMMARY ---")
    print(summary[0]['summary_text'])

except Exception as e:
    print("❌ Error:", e)
