# ✨ The Intelligent Data Preprocessor

An interactive web application built with Streamlit that automates, simplifies, and accelerates the data cleaning and preprocessing workflow. This tool transforms a raw CSV file into a clean, analysis-ready dataset in minutes, not hours.

---

### **Quick Links**

🎥 **Watch the Demo Video on YouTube**
<br>
[![Watch the video](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://YOUR-YOUTUBE-LINK)

🌐 **Try the Live Application**
<br>
[**[YOUR-DEPLOYMENT-LINK]**](https://YOUR-DEPLOYMENT-LINK)

---

## 🎯 Project Goal

The primary goal of this project is to minimize the cognitive load on the data analyst. By performing a deep, automated analysis of the data, the application intelligently identifies common issues and suggests best-practice solutions. The user is then empowered to review, adjust, and execute a complex preprocessing pipeline with confidence and ease.

This tool is designed to be a perfect blend of **intelligent automation** and **manual user control**.

## 🚀 Key Features

-   **Intelligent Analysis Engine:** Automatically scans the uploaded dataset for:
    -   Missing Values (counts and percentages).
    -   Duplicate Rows.
    -   Constant & Highly Correlated Columns.
    -   Potential ID / High-Cardinality Columns.
    -   Highly Empty Columns (>50% missing).
    -   Outliers in continuous numerical data.
    -   Data type inconsistencies (e.g., numbers stored as text).
-   **Smart UI Defaults:** The control panel is automatically pre-configured based on the analysis, pre-selecting columns for dropping and suggesting optimal cleaning strategies.
-   **Comprehensive Preprocessing Toolkit:** A wide range of cleaning and transformation options are available:
    -   **Normal Preprocessing:** Handle duplicates, missing values (with multiple strategies), rename columns, and strip whitespace.
    -   **Advanced DS Preprocessing:** Perform outlier capping (IQR/Z-score), feature scaling (MinMax/Standard), categorical encoding (One-Hot/Label), and low-variance feature removal.
    -   **NLP Preprocessing:** A full pipeline to clean text data, including lowercasing, stopword removal, lemmatization, and removal of URLs/special characters.
-   **Advanced Interactive Visualizations:** A data exploration playground that automatically selects the best chart (Histogram, Bar, Pie, Scatter, Box Plot, Heatmap) based on the data types of the selected columns.
-   **Transparent Logging:** A detailed and attractive log of every action taken, providing a clear audit trail of the entire cleaning process.

---

## 🛠️ Technology Stack

-   **Backend & Logic:** Python, Pandas, NumPy, Scikit-learn, NLTK
-   **Frontend & UI:** Streamlit
-   **Data Visualization:** Altair

---

## 📸 Application Showcase

### 1. The Interactive Analysis Report

Upon uploading a dataset, the application generates a multi-layered report.

-   **Health Report & Smart Suggestions:** The first section provides a high-level overview and highlights critical issues found in the data. The "Smart Suggestions" boxes clearly state what was found and what action has been pre-selected in the controls, bridging the gap between analysis and action.

    `![Analysis Report and Suggestions](screenshots/analysis-report.png)`

<br>

-   **Data Visualization Playground:** This is a powerful, interactive tool for data exploration. The application intelligently distinguishes between continuous numeric data and discrete/categorical data to provide the most insightful plot type.
    -   **Numeric vs. Categorical (`Age` vs. `Pclass`):** Automatically generates a Box Plot.
    -   **Numeric (`Age`):** Generates a Histogram with an intelligent default for the number of bins and an option to use a log scale for skewed data.

    `![Advanced Visualization Playground](screenshots/visualization-playground.png)`

### 2. The Controls Panel & Preprocessing

This is the command center where the user confirms or adjusts the tool's suggestions before running the pipeline.

-   **Intelligent Defaults:** The screenshot below shows how the tool has automatically pre-selected the `PassengerId` and `Cabin` columns for dropping and has pre-ticked the "Enable Outlier Capping" checkbox based on the initial analysis of the Titanic dataset.
-   **Full User Control:** While the defaults are smart, the user has complete manual control to change any setting before execution.

    `![Controls Panel with Smart Defaults](screenshots/controls-panel.png)`

### 3. The Final Output

After processing, the user can review the results and audit the entire workflow.

-   **Data Viewer:** A side-by-side comparison of the original and the final, cleaned DataFrame.
-   **Processing Log:** A clean, readable, and structured log detailing every single transformation applied to the data, complete with icons and collapsible sections.

    `![Final Output and Processing Log](screenshots/final-output.png)`

---

## 📖 How It Works: A Deeper Dive

<details>
<summary><strong>Click to expand: Detailed breakdown of the project's architecture</strong></summary>

The project is structured into several modular Python scripts, each with a specific responsibility.

-   **`app.py` - The Core Application:**
    -   This is the main script that runs the Streamlit web application.
    -   It handles the UI layout (tabs, columns, expanders), session state management, and the overall workflow logic.
    -   It contains the "execution engine" that gathers user selections from the UI and calls the appropriate backend functions in a specific, robust order.
    -   The advanced visualization logic, which intelligently selects the correct plot type based on data characteristics, is also located here.

-   **`analysis.py` - The Brains:**
    -   This module contains the `generate_full_analysis` function, which is the non-visual "brain" of the application.
    -   It takes a raw DataFrame and performs a deep statistical analysis to identify all the key data quality issues mentioned in the features list.
    -   It uses a series of heuristics (e.g., uniqueness ratio for ID detection, unique value counts for outlier filtering) to make its suggestions more context-aware and human-like.
    -   It returns a comprehensive dictionary (the "analysis report") that drives the entire UI.

-   **`normal_preprocessing.py` - The Basic Toolkit:**
    -   This module contains functions for fundamental data cleaning tasks.
    -   It relies heavily on efficient, built-in Pandas methods like `.drop_duplicates()`, `.fillna()`, and `.rename()`.
    -   Each function takes a DataFrame and a report list, performs its operation, appends a structured log entry to the report, and returns the modified DataFrame.

-   **`ds_preprocessing.py` - The Advanced Toolkit:**
    -   This module leverages the power of Scikit-learn for more advanced, machine learning-oriented preprocessing.
    -   It handles tasks like feature scaling (`MinMaxScaler`, `StandardScaler`), categorical encoding (`LabelEncoder`), and feature selection (`VarianceThreshold`).
    -   This module contains important logic to handle edge cases, such as filling missing values in categorical columns before attempting to encode them, thus preventing common errors.

-   **`nlp_preprocessing.py` - The Text Specialist:**
    -   This module uses NLTK (Natural Language Toolkit) and Regular Expressions to provide a standard pipeline for cleaning text data.
    -   It handles common NLP tasks such as lowercasing, stopword removal, and lemmatization.

</details>

---

## 🏁 Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
    cd YOUR-REPOSITORY-NAME
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your terminal.)*

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

The application should now be open and running in your web browser.