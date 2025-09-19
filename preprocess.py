import spacy
from typing import Dict, Any, List
import re

nlp = spacy.load("en_core_web_sm") 

def clean_text(text: str) -> str:
    """Basic cleaning without removing necessary punctuation for summarization."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text

def extract_entities(text: str) -> List[Dict[str,str]]:
    """Return list of entities with label and text."""
    doc = nlp(text)
    ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return ents

def split_text_for_model(text: str, max_tokens_estimate: int = 800) -> List[str]:
    """Naive splitter by sentence ensuring chunks are not too long for summarizers.
       max_tokens_estimate is approximate; values around 600-1000 chars are typical."""
    doc = nlp(text)
    chunks = []
    current = []
    current_len = 0
    for sent in doc.sents:
        s = sent.text.strip()
        if not s:
            continue
        if current_len + len(s) > max_tokens_estimate and current:
            chunks.append(" ".join(current))
            current = [s]
            current_len = len(s)
        else:
            current.append(s)
            current_len += len(s)
    if current:
        chunks.append(" ".join(current))
    return chunks
