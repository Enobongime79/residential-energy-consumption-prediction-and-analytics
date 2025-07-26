import pandas as pd

# If it's comma-separated (or specify delimiter)
df = pd.read_csv(r"C:\Users\Issac\Documents\Data Engineering\final-year-project\forecasting-energy-consumption-LSTM\dataset\kaggle_data_1h.csv")



# Assuming you have a DataFrame named 'df'
# For demonstration purposes, let's create a sample DataFrame:

# Check the data type of each column
column_types = df.dtypes

print("Data types of each column:")
print(column_types)

# You can also iterate through them if you want more custom output:
print("\nDetailed column types:")
for column_name, dtype in column_types.items():
    print(f"Column '{column_name}': {dtype}")

# If you want to check the type of the index (as in your previous issue):
print(f"\nIndex type: {df.index.dtype}")