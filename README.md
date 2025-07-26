# 🏠 Energy Consumption Forecasting (LSTM)

This project is a machine learning application that predicts residential energy consumption in real-time using an LSTM model. The app is built with Python and runs via a simple Streamlit interface.

---

## 📌 Project Summary

This tool helps forecast energy consumption based on historical power usage data. It allows users to train a unique LSTM model tailored to their dataset and visualize predictions. Ideal for smart building scenarios or IoT environments.

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, TensorFlow/Keras  
- **Interface:** Streamlit  
- **Notebook Platform (earlier stage):** Google Colab  

---

## 📊 Dataset

The dataset was sourced from the [UCI ML Repository (via Kaggle)](https://www.kaggle.com/uciml/electric-power-consumption-data-set).  
It includes minute-level measurements of household electrical power consumption.

---

## 🧹 Data Preprocessing

- Handled missing values  
- Smoothed time series using exponential smoothing  
- Detected and removed outliers using standard deviation  
- Normalized values between 0 and 1  
- Resampled data to appropriate time intervals  

---

## 🧠 Model

- Built and trained a **Long Short-Term Memory (LSTM)** model using TensorFlow/Keras  
- Split the dataset into training, validation, and test sets  
- Evaluated prediction accuracy using visualizations and metrics  

---

## 🚀 How to Run

> ⚠️ A decently powerful system is recommended due to training resource demands.

1. **Clone the repository**

```bash
git clone https://github.com/Enobongime79/forecasting-energy-consumption-LSTM.git
cd forecasting-energy-consumption-LSTM
Install requirements

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
Upload your dataset (must follow the format used in the original dataset)

📷 Prediction Results
The app displays LSTM predictions for energy consumption across the next 24 hours using interactive charts.

📂 Folder Structure
bash
Copy
Edit
.
├── app.py                # Streamlit app entry point
├── data/                 # Folder to place your CSV dataset
├── model/                # Saved models (if any)
├── performance.py        # Script for model evaluation
├── preprocessing.py      # Data cleaning and prep logic
├── README.md             # You are here
└── requirements.txt      # Dependencies
📄 License
This project is licensed under the MIT License
Forked and customized from the original repo by Christos Chousiadas

🙋‍♂️ Author
Isaac Ime
B.Sc. Computer Science, Covenant University
GitHub | LinkedIn
