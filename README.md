# ✨ The Intelligent Data Preprocessor

An interactive web application built with Streamlit that automates, simplifies, and accelerates the data cleaning and preprocessing workflow. This tool transforms a raw CSV file into a clean, analysis-ready dataset in minutes, not hours.

---

### **Quick Links**

🎥 **Watch the Demo Video on YouTube**
<br>
[![Watch the video](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://YOUR-YOUTUBE-LINK)

🌐 **Try the Live Application**
<br>
[**https://preprocess-your-data.streamlit.app/**](https://preprocess-your-data.streamlit.app/)

---

## 🎯 Project Goal

The primary goal of this project is to minimize the cognitive load on the data analyst. By performing a deep, automated analysis of the data, the application intelligently identifies common issues and suggests best-practice solutions. The user is then empowered to review, adjust, and execute a complex preprocessing pipeline with confidence and ease.

This tool is designed to be a perfect blend of **intelligent automation** and **manual user control**.

## 🚀 Key Features & Technical Logic

This application is more than just a wrapper around libraries; it contains a set of custom heuristics and logic to make intelligent, context-aware decisions.

-   **Intelligent Analysis Engine:** The application automatically performs a deep scan of the dataset to identify potential issues before any cleaning begins.

    <details>
    <summary><strong>🧠 Behind the Scenes: The Analysis Logic</strong></summary>
    
    The analysis engine doesn't just calculate basic statistics; it uses a series of rules to find actionable insights:
    -   **Potential ID Detection:** A column is flagged as a potential ID if it meets a two-part heuristic:
        1.  Its uniqueness ratio (`unique_values / total_rows`) is greater than 95%.
        2.  OR, its name contains a keyword like `id`, `no`, or `key`, and its uniqueness is still very high (>85%).
    -   **Outlier Detection:** Outliers are calculated using the standard **IQR (Interquartile Range)** method. However, to avoid incorrectly flagging discrete counts (like a 'Rating' column from 1-5), this check is **only performed on numeric columns with more than 20 unique values**, treating them as truly continuous data.
    -   **Highly Empty Column Detection:** Any column with more than **50% missing values** is flagged as "highly empty," as imputation is often not a viable strategy in such cases.
    -   **Correlation Analysis:** The Pearson correlation matrix is calculated for all continuous numeric columns, and any pair with a correlation coefficient **greater than 0.95** is flagged for potential multicollinearity.
    
    </details>

-   **Smart UI Defaults:** The control panel is automatically pre-configured based on the analysis, guiding the user toward best practices.

    <details>
    <summary><strong>⚙️ Behind the Scenes: The Pre-selection Logic</strong></summary>

    The "smart" part of the UI comes from mapping the analysis findings directly to the `default` parameters of the Streamlit widgets:
    -   The **"Columns to Remove"** multiselect is pre-populated with any columns flagged as **Potential IDs** and **Highly Empty Columns**.
    -   The **"Remove Constant Columns"** and **"Remove Duplicate Rows"** checkboxes are automatically ticked if the analysis finds them.
    -   The **"Enable Outlier Capping"** checkbox is automatically ticked, and the relevant columns are pre-selected in the multiselect if the analysis flags any significant outliers.
    -   **NLP options** are pre-selected based on a "best practice" baseline (lowercase, remove stopwords, etc.), with `remove_urls` being intelligently added only if URLs are detected in the text.
    
    </details>

-   **Advanced Interactive Visualizations:** A data exploration playground that automatically selects the best chart type and provides intelligent plotting options.

    <details>
    <summary><strong>📊 Behind the Scenes: The Visualization Logic</strong></summary>

    The playground avoids being just a simple plotting tool by incorporating contextual intelligence:
    -   **Plot Type Selection:** The logic for choosing a chart is nuanced:
        1.  A column is first classified as **"Continuous Numeric"** (numeric type with >20 unique values) or **"Categorical/Discrete"** (object type or numeric with <=20 unique values).
        2.  This classification then determines the plot:
            -   `Continuous` vs. `Continuous` -> **Scatter Plot**
            -   `Continuous` vs. `Categorical/Discrete` -> **Box Plot**
            -   `Categorical/Discrete` vs. `Categorical/Discrete` -> **Heatmap**
    -   **Intelligent Defaults:**
        -   **Histogram Bins:** The default number of bins is calculated using the robust **Freedman-Diaconis rule**, providing a better starting point than a fixed number.
        -   **Log Scale:** The "Use Log Scale" checkbox is automatically ticked for highly skewed data (skewness > 2), making skewed distributions instantly more readable.

    </details>

-   **Comprehensive Preprocessing Toolkit:** A wide range of options for normal cleaning, advanced data science transformations, and NLP text processing.

-   **Transparent Logging:** A detailed and attractive log of every action taken, providing a clear audit trail of the entire cleaning process.

---

## 📸 Application Showcase

The application workflow is designed to be intuitive, guiding the user from initial analysis to the final, cleaned dataset. Here’s a walkthrough of the key steps:

### 1. Instant Analysis & Smart Suggestions

Upon uploading a dataset, the application immediately performs a deep analysis and presents a multi-layered report. This is the starting point for all data cleaning activities.

-   **The Health Report** provides a high-level overview, while the **Smart Suggestions** highlight critical issues that the tool has automatically identified. This bridges the gap between analysis and action by explaining *what* was found and *why* a certain action has been pre-selected in the controls.

<p align="center">
  <img src="screenshots/Smart_Suggestions.png" alt="Analysis Report and Suggestions" width="800px">
</p>

-   The **Column Summary Table** gives a dense, sortable overview of every column, with visual heatmaps to quickly identify missing data and high-cardinality features. This allows for rapid identification of problematic columns.

<p align="center">
  <img src="screenshots/Summary.png" alt="Column Summary Table" width="800px">
</p>

### 2. Intelligently Pre-configured Controls

After reviewing the analysis, the user moves to the control panel. The true power of the tool lies in its ability to translate the analysis into tangible actions by intelligently pre-configuring the UI.

-   Here, the tool has correctly identified `PassengerId` and `Name` as potential ID columns and has **pre-selected them for removal**. The user can then accept or override this suggestion, maintaining full control.

<p align="center">
  <img src="screenshots/normal.png" alt="Controls Panel with Smart Defaults" width="500px">
</p>

-   The same intelligence applies to all sections, including the **NLP Preprocessing** module. It automatically detects text columns and suggests a robust set of default cleaning operations to save the user time.

<p align="center">
  <img src="screenshots/NLP.png" alt="NLP Preprocessing Panel" width="500px">
</p>

### 3. Advanced Visualization Playground

Before or after cleaning, the user can use the playground for deep, interactive data exploration. The application intelligently selects the best chart type to reveal insights.

-   The playground can create **univariate plots**, offering a choice between Bar and Pie charts for low-cardinality data, allowing for flexible analysis of single variables.

<p align="center">
  <img src="screenshots/Visualisation-1.1.png" alt="Pie Chart Visualization" width="800px">
</p>

-   It excels at **bivariate analysis**. When comparing two numeric columns, it automatically generates a Scatter Plot to help the user visually identify correlations and clusters.

<p align="center">
  <img src="screenshots/Visualisation-2.png" alt="Scatter Plot Visualization" width="800px">
</p>

- When comparing two Categorical columns, it automatically generates a heatmap to help the user visually.

<p align="center">
  <img src="screenshots/Visualisation-2.2.png" alt="heatmap" width="800px">
</p>

-   The tool is also smart enough to prevent user error, blocking attempts to visualize high-cardinality ID columns that would produce a meaningless chart.

<p align="center">
  <img src="screenshots/Visualisation-0.png" alt="ID Column Blocking Feature" width="800px">
</p>

### 4. Final Review and Audit

Once the user is satisfied with their selections and clicks "Run Preprocessing", they can review the results and audit the entire workflow.

-   The **Data Viewer** provides a clear side-by-side comparison of the original and the final, cleaned DataFrame. This screenshot shows the result of running the NLP cleaner on a dataset, making the impact of the transformation immediately obvious.

<p align="center">
  <img src="screenshots/Transformed.png" alt="Final Output in Data Viewer" width="800px">
</p>

-   The **Processing Log** generates a clean, readable, and structured report detailing every single transformation applied to the data, ensuring complete transparency and reproducibility.

<p align="center">
  <img src="screenshots/log.png" alt="Processing Log" width="800px">
</p>

---

## 📖 How It Works: A Deeper Dive

<details>
<summary><strong>Click to expand: Detailed breakdown of the project's architecture</strong></summary>

The project is structured into several modular Python scripts, each with a specific responsibility.

-   **`app.py` - The Core Application:** Runs the Streamlit web app, handles the UI layout, session state, and the overall workflow logic. It contains the "execution engine" and the advanced visualization logic.

-   **`analysis.py` - The Brains:** Contains the `generate_full_analysis` function. It takes a raw DataFrame, performs a deep statistical analysis using a series of heuristics (e.g., uniqueness ratio for ID detection), and returns a comprehensive dictionary that drives the entire UI.

-   **`normal_preprocessing.py` - The Basic Toolkit:** Contains functions for fundamental data cleaning tasks, relying on efficient, built-in Pandas methods.

-   **`ds_preprocessing.py` - The Advanced Toolkit:** Leverages Scikit-learn for more advanced, machine learning-oriented preprocessing like scaling, encoding, and feature selection.

-   **`nlp_preprocessing.py` - The Text Specialist:** Uses NLTK and Regular Expressions to provide a standard pipeline for cleaning text data.

</details>

---

## 💡 Future Improvements

This project provides a strong foundation for an even more powerful data processing tool. Potential future enhancements include:

-   **Target Variable Analysis:** Allow the user to select a "target" column, which would unlock specific analyses like class imbalance reports (for classification) or feature correlation heatmaps against the target (for regression).
-   **Advanced Encoding Methods:** Implement more sophisticated categorical encoding techniques, such as **Target Encoding** or **Feature Hashing**, to better handle high-cardinality features.
-   **Date/Time Feature Engineering:** Automatically detect `datetime` columns and provide options to extract valuable features like year, month, day of the week, or "is_weekend".
-   **Save & Load a Pipeline:** Allow users to save their sequence of selected operations and settings. This would enable them to instantly re-apply the exact same cleaning process to new data with the same schema.
-   **"Generate Python Code" Button:** A feature to generate and display the equivalent `pandas` and `scikit-learn` Python code for the selected operations, turning the tool into a powerful educational resource.

---

## 🏁 Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DarshitaDwivedii/Preprocess-Your-Data.git
    cd Preprocess-Your-Data
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

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

The application should now be open and running in your web browser.