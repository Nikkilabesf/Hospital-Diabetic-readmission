🌸 Hospital Diabetic Readmission Prediction

💖 Using data + machine learning to help hospitals care smarter — and save lives.



💡 Overview

This project uses machine learning to predict which diabetic patients are most likely to be readmitted to the hospital within 30 days. The goal is to help healthcare providers take preventive action — improving outcomes, saving costs, and empowering patients.

Think of it as: data meets compassion. 🩺📊

🎯 Motivation

Hospital readmissions are expensive — not just in dollars, but in patient well-being. Diabetic patients often struggle with complications that lead to repeat hospital stays.
By predicting who’s at risk before it happens, hospitals can step in earlier with care plans, follow-ups, and education.

💬 "Predictive analytics with heart."

🧬 Dataset

Source: UCI Machine Learning Repository – Diabetes 130-US Hospitals

Target: Readmission within 30 days (Yes / No)

Key Features:

Age, Gender, Race

Number of Procedures, Lab Tests, Medications

Diagnosis Codes (ICD)

Hospital Stay Length, Discharge Disposition

Data Prep Steps:
🧹 Clean missing values
🏷️ Encode categorical columns
🧠 Feature engineering (visit counts, med changes)
⚖️ Handle class imbalance (SMOTE/weighting)

💻 Tech Stack & Tools
Area	Tools Used
Language	🐍 Python 3
Data Wrangling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Modeling	Scikit-learn, XGBoost
Interpretability	SHAP, Feature Importances
Version Control	Git + GitHub (✨ of course ✨)
📊 Methodology

Exploratory Data Analysis (EDA) – understand patterns, plot correlations 💫

Feature Engineering – turn raw data into meaningful signals (e.g., prior visits)

Model Training – tested Logistic Regression, Random Forest, XGBoost

Evaluation Metrics – AUC-ROC, F1, Precision, Recall 💖

Explainability – used SHAP values to understand what features matter most

🌷 Results Highlights

✨ Best Model: XGBoost with AUC ≈ 0.83 and Recall ≈ 0.69
✨ Top Predictors: Number of Prior Admissions, Discharge Disposition, Length of Stay
✨ Insight: Patients with multiple previous stays + insulin changes were most likely to return soon

🩺 This means providers can flag high-risk patients and act before readmission occurs.

🪞 Visuals Include

Heatmaps of Feature Correlations

Class Distribution Charts

Confusion Matrix

Training vs Validation Loss Curves

Feature Importance Bar Charts

🩰 (All plots saved in /notebooks/plots)

⚙️ How to Run It
# Clone the repo
git clone https://github.com/Nikkilabesf/Hospital-Diabetic-readmission.git
cd Hospital-Diabetic-readmission

# Install dependencies
pip install -r requirements.txt

# Run the main notebook
jupyter notebook notebooks/diabetic_readmission.ipynb


Output plots + model saved under /models and /notebooks/plots.




Hospital-Diabetic-readmission/
│
├── data/               # Raw + Processed CSV Files  
├── models/             # Saved Machine Learning Models  
├── notebooks/          # EDA & Training Notebooks  
├── plots/              # Visualizations ✨  
├── requirements.txt    # Dependencies  
└── README.md           # You’re here 💕  



💕 Future Plans

Integrate Flask API for real-time risk predictions

Deploy dashboard for hospitals (Streamlit / Gradio)

Test with synthetic EHR data for broader validation

Optimize recall without sacrificing precision





🤝 Contributing

Want to collab on something that combines health & tech with a touch of ✨ femininity ✨?
Fork the repo, make your branch, and submit a PR. All clean, respectful energy welcome.

📜 License

Licensed under the MIT License – free to use, share, and improve.

💌 Connect with Me

👩🏽‍💻 Tenika Powell (Nikki Labesf)
📫 LinkedIn
 | GitHub

🩷 “Coding to change the future — one model at a time.”
