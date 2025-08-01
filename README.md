# 📈 Stock Price Prediction Using Machine Learning

This repository contains a machine learning project developed as part of my internship at **Developers Hub** in the **Artificial Intelligence and Machine Learning** domain. The goal of the project is to predict a stock’s next-day closing price using historical market data.

---

## 🎯 Objective

To build a regression-based machine learning model that predicts the next day’s **closing price** of a stock using features such as **Open**, **High**, **Low**, and **Volume**.

---

## 🗂️ Dataset

- **Source**: [Yahoo Finance](https://finance.yahoo.com/)
- **Access Method**: `yfinance` Python library
- **Sample Stock**: Apple Inc. (`AAPL`) *(You may change this to Tesla, Amazon, etc.)*
- **Features Used**:
  - Open
  - High
  - Low
  - Volume
- **Target Variable**:
  - Close (next-day prediction)

---

## 🛠️ Tools & Technologies

- Python  
- pandas & numpy (data manipulation)  
- matplotlib & seaborn (data visualization)  
- yfinance (data retrieval)  
- scikit-learn (modeling and evaluation)

---

## 🚀 Project Workflow

1. **Data Collection**  
   - Historical stock data was fetched using the `yfinance` API.
  
2. **Data Preprocessing**  
   - Handled missing values and aligned the target variable for next-day prediction.
  
3. **Feature Engineering**  
   - Used existing features and shifted the target column to predict future values.
  
4. **Model Training**  
   - Trained models using:
     - Linear Regression  
     - Random Forest Regressor

5. **Model Evaluation**  
   - Visual comparison of actual vs predicted prices  
   - Metrics used: R² Score, Mean Squared Error (MSE)

---

## 📊 Visualizations

- Line Plot: Actual vs Predicted Close Prices  
- Feature Correlation Heatmap  
- Model Residuals Plot

---

## 🧠 Skills Demonstrated

- Time Series Data Handling  
- Real-World Data Retrieval using APIs  
- Regression Modeling  
- Financial Forecasting  
- Data Visualization  
- Evaluation of Predictive Models

---

## ✅ Outcome

Successfully built and evaluated models capable of forecasting short-term stock prices with reasonable accuracy. This project enhanced my understanding of time-series modeling, financial data patterns, and real-time prediction systems — all within the context of AI and machine learning.

---

## 📌 Internship Credit

This project was developed under the guidance of mentors at **Developers Hub** as part of a comprehensive internship program in **AI and ML**, aimed at providing hands-on experience with real-world datasets and predictive modeling techniques.

---
