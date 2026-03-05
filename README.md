# ❤️ Heart Disease Prediction System

A Machine Learning powered web application that predicts the likelihood of heart disease using patient health attributes.
The system analyzes medical parameters and provides real-time predictions through an interactive web interface built with Streamlit.

## 📑 Table of Contents

- [Project Overview](#Project-Overview)
- [Problem Statement](#Problem-Statement)
- [Dataset Description](#Dataset-Description)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Feature Engineering](#Feature-Engineering)
- [Model Building & Comparison](#Model-Building--Comparison)
- [Web Application](#-Web-Application)
- [Project Structure](#-Project-Structure)
- [Technologies Used](#-Technologies-Used)
- [Installation Guide](#-Installation-Guide)
- [Usage](#-Usage)
- [Results](#-Results)
- [Learning & Conclusion](#-Learning--Conclusion)
- [Author](#--Author)
  
## 📊 Project Overview

Heart disease is one of the leading causes of death worldwide, and early prediction can help reduce the risk of severe health complications.

This project applies Machine Learning techniques to analyze patient health records and predict the presence of heart disease.

The system performs:

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

Model training and evaluation

Deployment of an interactive web app

The final system allows users to input medical parameters and receive an instant prediction of heart disease risk.

## ❗ Problem Statement

Traditional diagnosis of heart disease often requires multiple medical tests and clinical analysis.

This project aims to build a machine learning model capable of predicting whether a patient is likely to have heart disease based on medical attributes.

The target variable represents:

0 → No Heart Disease

1 → Heart Disease Present

The objective is to:

Identify important health indicators

Detect patterns within patient data

Predict heart disease risk for new patients

Assist healthcare professionals with data-driven insights

## 📂 Dataset Description

The dataset used in this project contains patient health records including attributes such as:

### Feature	Description

Age	Age of the patient
Sex	Gender (1 = Male, 0 = Female)
Chest Pain Type	Type of chest pain experienced
Resting Blood Pressure	Blood pressure level
Cholesterol	Serum cholesterol level
Fasting Blood Sugar	Blood sugar level
Rest ECG	Electrocardiographic results
Max Heart Rate	Maximum heart rate achieved
Exercise Angina	Exercise induced angina
Oldpeak	ST depression induced by exercise
Slope	Slope of ST segment
CA	Number of major vessels
Thal	Thalassemia status
Target	Presence of heart disease

### 🧹 Data Preprocessing

Before applying machine learning models, the dataset was cleaned and prepared.

### Steps performed:

Handling missing values

Standardizing column names

Encoding categorical variables

Feature scaling using StandardScaler

Splitting data into training (80%) and testing (20%)

These steps ensured the dataset was ready for reliable model training.

## 📈 Exploratory Data Analysis

Exploratory Data Analysis helped understand patterns and relationships within the dataset.

Main techniques used:

Correlation heatmaps

KDE distribution plots

Violin plots

Pair plots

Key Findings

Important indicators of heart disease:

Chest pain type

Maximum heart rate

ST depression (oldpeak)

Age group (40–55)

Weaker indicators:

Cholesterol

Resting blood pressure

## ⚙️ Feature Engineering

To improve model performance, feature engineering techniques were applied.

These included:

Encoding categorical variables

Scaling numerical features

Feature selection using correlation analysis

Important predictive features identified:

Age

Chest pain type

Maximum heart rate

Oldpeak

Slope

## 🤖 Model Building & Comparison

Three machine learning models were evaluated:

Model	Performance
Logistic Regression	Best overall performance
Random Forest	Strong ROC-AUC performance
XGBoost	Moderate performance
Best Model

### Logistic Regression

Performance:

Accuracy ≈ 80%

Recall ≈ 85%

ROC-AUC ≈ 0.87

This model provided the most balanced and reliable predictions.

## 🌐 Web Application

An interactive web application was developed using Streamlit.

The app allows users to:

### 1️⃣ Dataset Overview

View dataset preview, summary statistics, and missing values.

### 2️⃣ Exploratory Data Analysis

Interactive visualizations including heatmaps and distributions.

### 3️⃣ Model Evaluation

Displays evaluation metrics:

Accuracy

Precision

Recall

F1 Score

ROC Curve

Confusion Matrix

### 4️⃣ Prediction Interface

Users can input health parameters and receive an instant heart disease prediction.

## 📁 Project Structure
heart-disease-prediction
│
├── data
│   └── heart.csv
│
├── notebook
│   └── heart_disease_analysis.ipynb
│
├── model
│   └── trained_model.pkl
│
├── heart_app_streamlit.py
├── requirements.txt
└── README.md

## 🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Streamlit

## ⚡ Installation Guide

Clone the repository

git clone https://github.com/yourusername/heart-disease-prediction.git

Navigate to the project folder

cd heart-disease-prediction

Install dependencies

pip install -r requirements.txt

Run the Streamlit app

streamlit run heart_app_streamlit.py
## ▶️ Usage

Open the web application

Enter patient health parameters

Click Predict

The system displays the probability of heart disease

## 📊 Results

The project successfully demonstrated how machine learning can be used to predict heart disease risk.

Key outcomes:

Logistic Regression achieved the best results

Important features influencing prediction were identified

A user-friendly web app was deployed for real-time predictions

## 🎓 Learning & Conclusion

This project highlighted several important lessons:

Data exploration is critical for understanding patterns

Simpler models can outperform complex models

Feature importance improves model interpretability

Deploying models improves accessibility for users

Overall, the project demonstrates the potential of data science in healthcare decision support systems.

## 👨‍💻 Author

 Balram Shah

 GitHub: [(your GitHub link)](https://github.com/balram-shah)

## ⭐ If you found this project useful, consider giving it a star on GitHub
