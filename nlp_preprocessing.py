# nlp_preprocessing.py
import pandas as pd
import re
import nltk

# --- Self-contained downloader function ---
def _ensure_nltk_data_is_downloaded():
    """
    Private function to check for all necessary NLTK data and download if missing.
    """
    # Dictionary of resources to check and download. Key is download name, value is find path.
    resources = {
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
        "stopwords": "corpora/stopwords",
        "punkt_tab": "tokenizers/punkt_tab" # <-- ADDED THIS LINE TO FIX THE ERROR
    }

    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

# --- Call the downloader function once when the module is imported ---
_ensure_nltk_data_is_downloaded()

# Now we can safely import the rest
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def process_text_column(df, report, text_col, funcs_to_apply):
    """Master function to apply selected NLP steps to a text column."""
    if not text_col or text_col not in df.columns:
        report.append("✗ NLP Warning: No valid text column selected or found.")
        return df, report

    df[text_col] = df[text_col].astype(str)
    report.append(f"🔹 **NLP Processing on column: '{text_col}'**")
    
    # The rest of your function is perfect and remains unchanged
    if 'lowercase' in funcs_to_apply:
        df[text_col] = df[text_col].str.lower()
        report.append(f"  - Converted text to lowercase.")
        
    if 'remove_urls' in funcs_to_apply:
        df[text_col] = df[text_col].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
        report.append(f"  - Removed URLs.")

    if 'remove_special_chars' in funcs_to_apply:
        df[text_col] = df[text_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        report.append(f"  - Removed special characters.")
        
    if 'remove_stopwords' in funcs_to_apply:
        stop_words = set(stopwords.words('english'))
        df[text_col] = df[text_col].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
        report.append(f"  - Removed stopwords.")

    if 'lemmatize' in funcs_to_apply:
        lemmatizer = WordNetLemmatizer()
        df[text_col] = df[text_col].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
        report.append(f"  - Lemmatized text.")
        
    return df, report