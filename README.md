# 🌱 Sustainable Fertilizer Recommendation System Using Python

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-brightgreen?style=for-the-badge&logo=render)](https://sustainable-fertilizer-recommendation.onrender.com/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://sustainable-fertilizer-recommendation.onrender.com/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

A complete, beginner-friendly Machine Learning and Web Application designed to help farmers and agricultural students determine the most suitable and eco-friendly fertilizer recommendations based on soil nutrients (Nitrogen, Phosphorus, Potassium), soil pH, and crop variety.

🌐 **Live Deployed Web App:** [https://sustainable-fertilizer-recommendation.onrender.com/](https://sustainable-fertilizer-recommendation.onrender.com/)

---

## 📋 Table of Contents
1. [Live Application URL](#-live-deployed-web-app)
2. [Project Overview](#-project-overview)
3. [Key Features](#-key-features)
4. [Technology Stack](#-technology-stack)
5. [Project File Structure](#-project-file-structure)
6. [How Each File Works](#-how-each-file-works)
7. [Installation & Setup Guide](#-installation--setup-guide)
8. [Running the Application](#-running-the-application)
9. [Machine Learning Details](#-machine-learning-details)
10. [Agronomic Knowledge & Rules](#-agronomic-knowledge--rules)
11. [College Viva & Demonstration Q&A](#-college-viva--demonstration-qa)
12. [Disclaimer](#-disclaimer)

---

## 🌾 Project Overview
Overuse and improper application of chemical fertilizers leads to severe soil degradation, groundwater nitrate contamination, eutrophication of rivers and lakes, and elevated agricultural costs. 

The **Sustainable Fertilizer Recommendation System** solves this challenge by analyzing:
- **Nitrogen (N)** level
- **Phosphorus (P)** level
- **Potassium (K)** level
- **Soil pH** (acidity / alkalinity)
- **Crop Type** (Rice, Wheat, Maize, Cotton, Sugarcane, Tomato, Potato, Groundnut)

Using a **Decision Tree Classifier** from `scikit-learn`, the system identifies soil nutrient deficiencies and suggests targeted fertilizers (such as **Urea, DAP, MOP, NPK complexes, Compost, or Organic Manure**), paired with sustainable usage tips and environmental impact summaries.

---

## ✨ Key Features
- **Intuitive Web UI:** Agriculture-themed green-and-white interface built with Streamlit.
- **Interactive Soil Diagnostic Cards:** Classifies N, P, K levels into *Low*, *Optimal*, or *High*, and pH into *Acidic*, *Neutral*, or *Alkaline*.
- **Decision Tree ML Backend:** Transparent, highly interpretable machine learning model (~96% accuracy on test data).
- **Sustainable Farming Tips:** Guides farmers on the *4R Nutrient Stewardship* (Right Source, Right Rate, Right Time, Right Place), split dosing, and organic amendments.
- **Environmental Benefits Breakdown:** Explains how the recommendation prevents water pollution and soil acidification.
- **Model & Dataset Explorer Tab:** Interactive feature importance bar chart and sample data browser for classroom and viva presentations.
- **One-Click Sidebar Demonstration Presets:** Easily test scenarios like *Low Nitrogen*, *Low Phosphate*, *Low Potash*, *Acidic Soil*, or *Healthy Soil* with a single click.

---

## 🛠️ Technology Stack
- **Programming Language:** Python (3.9+)
- **Frontend / Web UI:** [Streamlit](https://streamlit.io/)
- **Machine Learning Library:** [Scikit-learn](https://scikit-learn.org/) (Decision Tree Classifier)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- **Persistence:** Python `pickle`
- **Database:** None required (reads lightweight CSV dataset)
- **External APIs:** None required (100% offline, local execution)

---

## 📁 Project File Structure
```text
hans/
│
├── app.py                  # Main Streamlit web application
├── model.py                # Machine learning training, evaluation & prediction logic
├── fertilizer_data.csv     # Demonstration dataset with 160 soil & crop profiles
├── fertilizer_model.pkl    # Serialized trained model & encoders artifact
├── requirements.txt        # List of required Python dependencies
└── README.md               # Complete project documentation & viva guide
```

---

## 🔍 How Each File Works

### 1. `app.py`
- Serves as the presentation layer using **Streamlit**.
- Provides a clean sidebar with one-click presets and demonstration disclaimers.
- Features three tabs:
  - **🌱 Fertilizer Recommendation:** Input fields for N, P, K, pH, and crop selection; displays diagnostic metric cards and recommendation summaries.
  - **📊 Model & Dataset Explorer:** Displays model accuracy, feature importance bar chart, and sample rows from the CSV dataset.
  - **📚 Sustainable Agriculture Guide:** Educational guide on N-P-K functions, soil pH availability, and the 4R Stewardship Framework.
- Applies custom CSS for a modern, responsive green-and-white theme.

### 2. `model.py`
- Contains core machine learning and agronomic evaluation functions:
  - `evaluate_soil_nutrients()`: Classifies soil N, P, K values (Low, Optimal, High) and pH (Acidic, Neutral, Alkaline).
  - `train_model()`: Encodes categorical features, trains `DecisionTreeClassifier(max_depth=6)`, evaluates accuracy, and exports `fertilizer_model.pkl`.
  - `load_or_train_model()`: Caches the trained model or retrains if missing.
  - `predict_fertilizer()`: Runs inference on user inputs and enriches predictions with sustainable application guidance and environmental benefits from `FERTILIZER_KNOWLEDGE_BASE`.
- Can be run standalone from terminal (`python model.py`) to inspect training metrics.

### 3. `fertilizer_data.csv`
- A demonstration dataset containing 160 balanced samples across 8 crop varieties:
  - **Inputs:** `Nitrogen`, `Phosphorus`, `Potassium`, `pH`, `Crop_Type`
  - **Target:** `Fertilizer` (Urea, DAP, MOP, 10-26-26, 14-35-14, 20-20-20, Compost, Organic Manure)
- Explicitly labeled as **demonstration data** for educational purposes.

### 4. `requirements.txt`
- Specifies the required packages: `pandas`, `numpy`, `scikit-learn`, `streamlit`.

---

## 🚀 Installation & Setup Guide

### Step 1: Open Terminal or Command Prompt
Navigate to the project root folder:
```bash
cd c:\Users\hp\Downloads\hans
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
Run:
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Launch the Streamlit Web Application
Execute the following command:
```bash
streamlit run app.py
```
Streamlit will automatically open your default browser at:
```text
http://localhost:8501
```

### Option B: Test the Machine Learning Model Standalone
To inspect model training, accuracy score, and feature importances via the terminal:
```bash
python model.py
```

---

## 🧠 Machine Learning Details

### Why Decision Tree Classifier?
1. **High Interpretability:** Decision trees mirror human decision-making and agronomic flowcharts (e.g. *If Nitrogen < 30 and pH between 6.0 and 7.5, then Urea*).
2. **Handles Non-Linear Relationships:** Soil nutrient dynamics and crop requirements are non-linear; decision trees naturally segment the feature space without complex transformations.
3. **Low Computational Overhead:** Trains in less than 0.1 seconds and requires negligible memory, making it ideal for local deployment without dedicated GPUs.
4. **No Standard Scaling Needed:** Unlike Support Vector Machines or Neural Networks, Decision Trees are invariant to feature scaling (e.g., pH ranges from 0–14 while N ranges from 0–250).

### Features Used:
- `Nitrogen` (Numeric, kg/ha or ppm)
- `Phosphorus` (Numeric, kg/ha or ppm)
- `Potassium` (Numeric, kg/ha or ppm)
- `pH` (Numeric, 0.0 – 14.0)
- `Crop_Type` (Categorical, encoded via `LabelEncoder`)

---

## 🌿 Agronomic Knowledge & Rules

| Fertilizer | Primary Nutrients | Best Used When... | Sustainable Benefit |
| :--- | :--- | :--- | :--- |
| **Urea** | 46% Nitrogen (46-0-0) | Nitrogen is deficient (N < 30), P & K adequate | Split-dosing prevents nitrate leaching into groundwater |
| **DAP** | 18% N, 46% P₂O₅ (18-46-0) | Phosphorus is deficient (P < 20) | Root-zone placement prevents phosphate runoff & algal blooms |
| **MOP** | 60% K₂O (0-0-60) | Potassium is deficient (K < 20) | Enhances plant water-use efficiency and drought resilience |
| **10-26-26** | Complex NPK (10-26-26) | Both P and K are moderately deficient | Integrated granule ensures uniform distribution without multiple runs |
| **14-35-14** | High P Complex (14-35-14) | High early phosphorus demand with starter N & K | Promotes rapid seedling root establishment |
| **20-20-20** | Balanced NPK (20-20-20) | General, balanced depletion across N, P, and K | Prevents skewed nutrient ratios in the rhizosphere |
| **Compost** | Organic Matter + Microbes | Nutrients are balanced; maintaining soil life | Sequesters carbon, improves aeration and water-holding capacity |
| **Organic Manure** | Multi-nutrient + Humic Acid | Soil pH is acidic (< 5.8) or alkaline (> 7.8) | Buffers soil pH naturally and revives beneficial soil earthworms |

---

## 🎓 College Viva & Demonstration Q&A

Here are common questions college professors and examiners ask during project evaluations, along with model answers:

### Q1: What is the primary problem this project addresses?
> **Answer:** Farmers frequently over-apply synthetic chemical fertilizers (especially Urea) due to lack of accessible soil diagnostic advice. This leads to soil acidification, water pollution, and unnecessary financial costs. This project provides a transparent AI recommendation tool that matches fertilizers specifically to nutrient deficiencies and crop needs.

### Q2: Why did you choose a Decision Tree Classifier instead of Deep Learning or Random Forest?
> **Answer:** 
> 1. Soil nutrient thresholds follow strict diagnostic boundaries, which decision trees model directly.
> 2. Decision trees provide complete transparency (*white-box model*) where every split can be explained agronomically.
> 3. For an educational and demonstration project with ~160 samples, deep learning would heavily overfit and require unnecessary computational resources.

### Q3: What do the letters N, P, and K stand for, and why are they important?
> **Answer:** 
> - **N (Nitrogen):** Drives leaf and vegetative shoot growth and chlorophyll production.
> - **P (Phosphorus):** Essential for deep root establishment, energy transfer (ATP), and flowering.
> - **K (Potassium):** Regulates stomatal water movement, strengthens stalks, and builds disease/drought resistance.

### Q4: How does soil pH influence fertilizer efficiency?
> **Answer:** Soil pH measures hydrogen ion concentration. In strongly acidic soils (pH < 5.5), phosphorus binds with aluminum and iron, becoming unavailable to plants. In alkaline soils (pH > 7.5), phosphorus precipitates with calcium and micronutrients become unavailable. Fertilizers work best in neutral soils (pH 6.0 – 7.5).

### Q5: How is the categorical feature `Crop_Type` processed by the model?
> **Answer:** Scikit-learn's `LabelEncoder` maps each unique crop name (e.g., Rice $\rightarrow$ 5, Wheat $\rightarrow$ 7) into an integer representation before feeding it into the decision tree algorithm.

### Q6: What does the Gini Impurity metric measure in Decision Trees?
> **Answer:** Gini impurity measures the likelihood that a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset. A Gini score of 0.0 represents a pure leaf node where all samples belong to the same fertilizer class.

### Q7: Why do you avoid giving exact fertilizer kilogram quantities?
> **Answer:** Fertilizer dosage depends on region-specific factors like soil depth, bulk density, irrigation methods, and local climate. Giving generic kilogram quantities without verified local calibration can be misleading; therefore, the system provides qualitative recommendations and best-practice guidance while recommending certified soil test centers for exact bulk dosage.

---

## ⚠️ Disclaimer
This system uses a **demonstration dataset** for academic and prototyping purposes. Predictions should not be used as scientific or commercial agricultural prescriptions without verified local soil testing and consultation with accredited agricultural extension officers.

---

## 👨‍💻 Author & Acknowledgements
- Developed for **Academic Demonstration & Sustainable Agriculture Education**.
- Built using **Python, Streamlit, and Scikit-Learn**.
