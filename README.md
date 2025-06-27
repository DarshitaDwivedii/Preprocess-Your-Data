# ✨ The Intelligent Data Preprocessor ✨

A comprehensive Streamlit web application for cleaning, transforming, and enhancing datasets with a few clicks. This tool combines normal data wrangling, advanced data science preprocessing, and essential NLP cleaning into a single, modular, and user-friendly pipeline.

<!-- 
    TIP: Create a short GIF of you using the app and upload it here. It's the best way to show what your project does!
    You can use tools like Giphy Capture, ScreenToGif, or Kap.

![App Demo GIF](https://user-images.githubusercontent.com/26284904/182855198-72b2210a-c558-466d-95f7-f8e4e9f55e5e.gif) 
*(This is a placeholder GIF, replace it with your own!)*

---
-->

## 🚀 Key Features

*   **🧹 Normal Preprocessing:**
    *   Manually drop or rename columns.
    *   Strip leading/trailing whitespace from string columns.
    *   Remove duplicate rows and constant-value columns.
    *   Optimize data types for memory efficiency.
    *   Handle missing values with per-column strategies (Mean, Median, Mode, Constant, or Drop).

*   **🔬 Advanced Data Science Preprocessing:**
    *   Detect and handle outliers using IQR or Z-score methods.
    *   Scale numerical features with MinMaxScaler or StandardScaler.
    *   Encode categorical features intelligently (One-Hot for low cardinality, Label for high cardinality).
    *   Remove low-variance features to reduce noise.

*   **📝 Natural Language Processing (NLP):**
    *   Process text data by converting to lowercase.
    *   Remove URLs, special characters, and stopwords.
    *   Lemmatize text to its root form.

*   **🏗️ Modular & Professional Design:**
    *   Logic is cleanly separated into `normal_pp`, `ds_pp`, and `nlp_pp` modules.
    *   Robust file loading that automatically detects and handles common CSV encodings.
    *   A detailed, step-by-step report is generated for every operation performed.

## 🛠️ Technology Stack

This project is built with a modern Python data science stack:

![Python](https://img.shields.io/badge/python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.2-150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-1.26-013243.svg?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-0A9D9B.svg?style=flat)


## 📋 Project Structure

The project is organized into logical modules for clarity and maintainability.

.
├── .gitignore # Tells Git which files to ignore
├── README.md # This file
├── requirements.txt # List of Python packages needed for the project
├── setup.py # Downloads necessary NLTK data
│
├── app.py # The main Streamlit application file (UI)
│
├── normal_preprocessing.py # Module for basic data cleaning tasks
├── ds_preprocessing.py # Module for data science/ML-specific tasks
├── nlp_preprocessing.py # Module for text processing tasks
└── analysis.py # (Future) Module for automated data analysis


## 🗺️ Roadmap: Future Enhancements

This project is actively being developed. Here are some features planned for the future:

-   [ ] **Automated Analysis Tab:** Integrate the `analysis.py` module to provide users with an automated report *before* preprocessing, suggesting which steps to apply.
-   [ ] **Data Visualization:** Add a tab for generating charts (histograms, box plots, correlation heatmaps) on the raw and processed data.
-   [ ] **Support for More File Types:** Add support for uploading and processing Excel (`.xlsx`) and JSON (`.json`) files.
-   [ ] **Advanced NLP Features:** Implement more complex features like Named Entity Recognition (NER) or Topic Modeling.

