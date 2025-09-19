# summarizer.py
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List

def get_device():
    return 0 if torch.cuda.is_available() else -1

def build_summarizer(model_name: str = "facebook/bart-large-cnn"):
    # Use tokenizer+model for more control (optional)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if device==0 else -1)
    return summarizer

def summarize_text(summarizer, text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Summarize single piece of text with safety for length."""
    if not text.strip():
        return ""
    # call pipeline
    out = summarizer(text, max_length=max_length, min_length=min_length, truncation=True)
    return out[0]["summary_text"]

def summarize_long_text(summarizer, chunks: List[str], glue: str = " ", max_length=130, min_length=30) -> str:
    """Summarize each chunk and then combine & optionally summarize again."""
    partial_summaries = []
    for chunk in chunks:
        s = summarize_text(summarizer, chunk, max_length=max_length, min_length=min_length)
        partial_summaries.append(s)
    combined = glue.join(partial_summaries)
    # If combined is long, run one more pass:
    if len(combined.split()) > max_length * 2:
        final = summarize_text(summarizer, combined, max_length=max_length, min_length=min_length)
        return final
    return combined
