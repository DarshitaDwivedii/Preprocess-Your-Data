# ✨ The Intelligent Data Preprocessor

An interactive web application built with Streamlit that automates, simplifies, and accelerates the data cleaning and preprocessing workflow. This tool transforms a raw CSV file into a clean, analysis-ready dataset in minutes, not hours.

---

### **Quick Links**

🎥 **Watch the Demo Video on YouTube**
<br>
[![Watch the video](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://YOUR-YOUTUBE-LINK)

🌐 **Try the Live Application**
<br>
[**[YOUR-DEPLOYMENT-LINK]**](https://preprocess-your-data.streamlit.app/)

---

## 🎯 Project Goal

The primary goal of this project is to minimize the cognitive load on the data analyst. By performing a deep, automated analysis of the data, the application intelligently identifies common issues and suggests best-practice solutions. The user is then empowered to review, adjust, and execute a complex preprocessing pipeline with confidence and ease.

This tool is designed to be a perfect blend of **intelligent automation** and **manual user control**.

## 🚀 Key Features

-   **Intelligent Analysis Engine:** Automatically scans the uploaded dataset for missing values, duplicates, constant columns, potential IDs, highly empty columns, outliers, and more.
-   **Smart UI Defaults:** The control panel is automatically pre-configured based on the analysis, pre-selecting problematic columns for removal and suggesting optimal cleaning strategies.
-   **Comprehensive Preprocessing Toolkit:** A wide range of options for normal cleaning, advanced data science transformations (scaling, encoding), and NLP text processing.
-   **Advanced Interactive Visualizations:** A data exploration playground that automatically selects the best chart (Histogram, Bar, Scatter, Box Plot, Heatmap) based on the data types of the selected columns.
-   **Transparent Logging:** A detailed and attractive log of every action taken, providing a clear audit trail of the entire cleaning process.

---

## 🛠️ Technology Stack

-   **Backend & Logic:** Python, Pandas, NumPy, Scikit-learn, NLTK
-   **Frontend & UI:** Streamlit
-   **Data Visualization:** Altair

---

## 📸 Application Showcase

The application workflow is designed to be intuitive, guiding the user from analysis to action.

### 1. Instant Analysis & Smart Suggestions

Upon uploading a dataset, the application immediately generates a multi-layered report.

-   **The Health Report** provides a high-level overview, while the **Smart Suggestions** highlight critical issues found in the data, such as potential ID columns and significant outliers. This bridges the gap between analysis and action.

    <!-- Use the screenshot named: showcase-report.png -->
    ![Analysis Report and Suggestions](screenshots/Smart_Suggestions.png)

-   The **Column Summary Table** gives a dense, sortable overview of every column, with visual heatmaps to quickly identify missing data and high-cardinality features.

    <!-- Use the screenshot named: showcase-summary-table.png -->
    ![Column Summary Table](screenshots/Summary.png)

### 2. Intelligently Pre-configured Controls

The true power of the tool lies in its ability to translate analysis into action. The control panel is automatically configured based on the suggestions.

-   Here, the tool has identified `PassengerId` and `Name` as potential ID columns and has **pre-selected them for removal**. The user can then accept or override this suggestion.

    <!-- Use the screenshot named: showcase-controls.png -->
    ![Controls Panel with Smart Defaults](screenshots/normal.png)

-   The same intelligence applies to all sections, including the **NLP Preprocessing** module, which detects text columns and suggests a robust set of default cleaning operations.

    <!-- Use the screenshot named: showcase-nlp.png -->
    ![NLP Preprocessing Panel](screenshots/NLP.png)

### 3. The Advanced Visualization Playground

This is a powerful, interactive tool for deep data exploration. The application intelligently selects the best chart type for your analysis.

-   The playground can create **univariate plots**, offering a choice between Bar and Pie charts for low-cardinality data.

    <!-- Use the screenshot named: showcase-pie-chart.png -->
    ![Pie Chart Visualization](screenshots/Visualisation-1.1.png)

-   It excels at **bivariate analysis**, automatically generating a Scatter Plot for two numeric columns to reveal correlations and clusters.

    <!-- Use the screenshot named: showcase-scatter-plot.png -->
    ![Scatter Plot Visualization](screenshots/Visualisation-2.png)

-   The tool is also smart enough to prevent user error, blocking attempts to visualize high-cardinality ID columns that would produce a meaningless chart.

    <!-- Use the screenshot named: showcase-id-blocking.png -->
    ![ID Column Blocking Feature](screenshots/Visualisation-0.png)

### 4. Final Review and Audit

After running the pipeline, the user can review the results and audit the entire workflow.

-   The **Data Viewer** provides a clear side-by-side comparison of the original and the final, cleaned DataFrame. This screenshot shows the result of running the NLP cleaner on a dataset.

    <!-- Use the screenshot named: showcase-data-viewer.png -->
    ![Final Output in Data Viewer](screenshots/Transformed.png)

-   The **Processing Log** generates a clean, readable, and structured report detailing every single transformation applied to the data, ensuring complete transparency.

    <!-- Use the screenshot named: showcase-log.png -->
    ![Processing Log](screenshots/log.png)

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