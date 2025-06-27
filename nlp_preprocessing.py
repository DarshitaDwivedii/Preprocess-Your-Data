# nlp_preprocessing.py
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def process_text_column(df, report, text_col, funcs_to_apply):
    """Master function to apply selected NLP steps to a text column."""
    if not text_col or text_col not in df.columns:
        report.append("✗ NLP Warning: No valid text column selected or found.")
        return df, report

    # Ensure the column is of string type, handling potential NaN values
    df[text_col] = df[text_col].astype(str)
    
    report.append(f"🔹 **NLP Processing on column: '{text_col}'**")
    
    if 'lowercase' in funcs_to_apply:
        df[text_col] = df[text_col].str.lower()
        report.append(f"  - Converted text to lowercase.")
        
    if 'remove_urls' in funcs_to_apply:
        df[text_col] = df[text_col].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
        report.append(f"  - Removed URLs.")
    
    # Remove mentions and hashtags

    if 'remove_special_chars' in funcs_to_apply:
        # Made this case-insensitive to work even if lowercase isn't selected first
        df[text_col] = df[text_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        report.append(f"  - Removed special characters.")
        
    if 'remove_stopwords' in funcs_to_apply:
        # Tokenization is needed to check for stopwords
        stop_words = set(stopwords.words('english'))
        df[text_col] = df[text_col].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
        report.append(f"  - Removed stopwords.")

    if 'lemmatize' in funcs_to_apply:
        # Tokenization is needed for lemmatization
        lemmatizer = WordNetLemmatizer()
        df[text_col] = df[text_col].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
        report.append(f"  - Lemmatized text.")
        
    return df, report