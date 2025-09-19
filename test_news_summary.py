from newspaper import Article
from transformers import pipeline
url = "https://www.bbc.com/news/articles/c0q7l8ewj0wo"
try:
    article = Article(url) 
    article.download()
    article.parse()
    print("\n--- ARTICLE TITLE ---")
    print(article.title)
    print("\n--- ORIGINAL TEXT (first 500 chars) ---")
    print(article.text[:500])
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(article.text[:1000], max_length=150, min_length=50, do_sample=False)
    print("\n--- SUMMARY ---")
    print(summary[0]['summary_text'])
except Exception as e:
    print("❌ Error:", e)
