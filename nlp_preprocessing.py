import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def _ensure_nltk_data_is_downloaded():
    """Checks for all necessary NLTK data and downloads if missing."""
    # THIS DICTIONARY IS THE KEY. WE ADD THE MISSING 'punkt_tab'.
    resources = {
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
        "stopwords": "corpora/stopwords",
        "punkt_tab": "tokenizers/punkt_tab" # <-- THIS IS THE FIX
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

_ensure_nltk_data_is_downloaded()

def _create_log_entry(icon, title, details=None, category="NLP"):
    """Helper to create a structured log entry for NLP tasks."""
    return {"category": category, "icon": icon, "title": title, "details": details}

def process_text_column(df, report, text_col, funcs_to_apply):
    """Master function to apply selected NLP steps to a text column."""
    if not text_col or text_col not in df.columns: return df, report
    
    df[text_col] = df[text_col].astype(str)
    details = []
    
    if 'lowercase' in funcs_to_apply:
        df[text_col] = df[text_col].str.lower()
        details.append("Converted text to lowercase.")
    if 'remove_urls' in funcs_to_apply:
        df[text_col] = df[text_col].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
        details.append("Removed URLs.")
    if 'remove_special_chars' in funcs_to_apply:
        df[text_col] = df[text_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        details.append("Removed special characters.")
    if 'remove_stopwords' in funcs_to_apply:
        stop_words = set(stopwords.words('english'))
        df[text_col] = df[text_col].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
        details.append("Removed stopwords.")
    if 'lemmatize' in funcs_to_apply:
        lemmatizer = WordNetLemmatizer()
        df[text_col] = df[text_col].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
        details.append("Lemmatized text.")
        
    if details:
        report.append(_create_log_entry("📝", f"Processed Text in Column `{text_col}`", details))
    return df, report