# fetcher.py
from newspaper import Article
import feedparser
from typing import List, Dict

def fetch_article_from_url(url: str) -> Dict[str, str]:
    """Return dict with keys: title, text, authors, publish_date"""
    article = Article(url)
    article.download()
    article.parse()
    return {
        "title": article.title or "",
        "text": article.text or "",
        "authors": ", ".join(article.authors) if article.authors else "",
        "publish_date": article.publish_date.isoformat() if article.publish_date else "",
    }

def fetch_articles_from_rss(rss_url: str, limit: int = 5) -> List[Dict]:
    feed = feedparser.parse(rss_url)
    items = []
    for entry in feed.entries[:limit]:
        url = entry.link
        try:
            article = fetch_article_from_url(url)
            items.append({"url": url, "title": article["title"], "text": article["text"]})
        except Exception:
            # skip items that fail
            continue
    return items
