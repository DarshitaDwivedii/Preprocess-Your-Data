# setup.py
import nltk
import ssl

# This is a robust way to handle NLTK downloads, which can sometimes fail
# due to SSL certificate verification issues. This code attempts to create
# an unverified SSL context to bypass those errors.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # This is for older Python versions that don't have this attribute
    pass
else:
    # Overrides the default context with the unverified one
    ssl._create_default_https_context = _create_unverified_https_context

# --- Define ALL the resources we need ---
# We use a dictionary for clarity: key is the download name, value is the find path
resources = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "stopwords": "corpora/stopwords",
    "punkt_tab": "tokenizers/punkt_tab" # <-- ADDED THIS LINE TO FIX THE ERROR
}

print("--- Checking and Downloading NLTK Resources ---")

# --- Loop through the resources and download if missing ---
for name, path in resources.items():
    try:
        # Check if the resource is already present on the system
        nltk.data.find(path)
        print(f"[✔] Resource '{name}' is already downloaded.")
    except LookupError:
        # If the resource is not found, a LookupError is raised.
        # We catch it and initiate the download.
        print(f"[!] Resource '{name}' not found. Downloading...")
        nltk.download(name, quiet=True) # Using quiet=True for a cleaner log
        print(f"[✔] Successfully downloaded '{name}'.")

print("\n--- NLTK setup is complete. You can now run 'streamlit run app.py' ---")