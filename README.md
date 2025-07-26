

A machine learning application that predicts household energy consumption using Long Short-Term Memory (LSTM) networks. Built for time-series analysis, the app allows users to train a custom model on their dataset and visualize future consumption patterns — all through a simple Streamlit interface.

## 🚀 About the Project

This tool enables energy forecasting in smart building environments using deep learning. Users can upload historical energy usage data, train an LSTM model directly in the app, and view predictions for the next 24 hours.

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Pandas, NumPy, Matplotlib**
- **Streamlit** – for the user interface
- **Jupyter Notebook** (for experimentation & debugging)

## 📂 Dataset

The dataset used for this project comes from the [UCI Electric Power Consumption Dataset](https://www.kaggle.com/uciml/electric-power-consumption-data-set).

It contains household energy usage data recorded every minute over several years.

## ⚙️ Features

- Train an LSTM model on your **custom dataset**
- Preprocesses the data with:
  - Handling of missing values
  - Exponential smoothing
  - Outlier detection (via standard deviation)
  - Normalization to [0,1]
  - Resampling for temporal consistency
- Predict future energy usage (1h – 24h ahead)
- Visualize actual vs. predicted results

## 🖥 How to Run the App

> ⚠️ **Note:** Training deep learning models is resource-intensive. It is recommended to run this app on a machine with decent CPU and RAM (or GPU support for better performance).

1. **Clone the repo**

```bash
git clone https://github.com/Enobongime79/residential-energy-consumption-prediction-and-analytics.git
cd residential-energy-consumption-prediction-and-analytics
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

bash
Copy
Edit
streamlit run app.py
Upload your dataset and start forecasting!

Model Architecture
Single-layer LSTM network

Trained on sequences of normalized energy usage values

Model is retrained for each uploaded dataset to provide a unique, adaptive solution

Sample Results
Visual results display predicted vs. actual energy usage to give users feedback on model accuracy.

Future Improvements
Add option to choose between LSTM and other models (e.g., GRU, Seq2Seq)

Save trained models for reuse

License
MIT License
© 2024 Isaac Ime

Feel free to fork, contribute, or give this repo a ⭐ if you found it helpful!
