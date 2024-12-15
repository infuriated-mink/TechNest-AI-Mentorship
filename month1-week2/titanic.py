import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# Download the dataset from Kaggle
path = kagglehub.dataset_download("brendan45774/test-file")

# List all files in the downloaded dataset directory to find the correct file
print("Files in the dataset folder:")
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
for i, file in enumerate(files, start=1):
    print(f"{i}: {file}")

# Automatically choose the first CSV file if present
csv_files = [file for file in files if file.endswith('.csv')]
if csv_files:
    file_path = os.path.join(path, csv_files[0])
    print(f"\nUsing dataset file: {file_path}")
else:
    print("\nNo CSV file found in the dataset directory.")
    exit()

# Load the Titanic dataset
try:
    titanic_data = pd.read_csv(file_path)
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Display basic information about the dataset
print("Dataset Information:")
titanic_data.info()

# Preview the first few rows of the dataset
print("\nFirst 5 Rows of the Dataset:")
print(titanic_data.head())

# Clean and standardize the 'Sex' column
if 'Sex' in titanic_data.columns:
    titanic_data['Sex'] = titanic_data['Sex'].str.strip().str.lower()
    print("\nUnique values in 'Sex' after cleaning:")
    print(titanic_data['Sex'].unique())
else:
    print("\nColumn 'Sex' not found in dataset.")
    exit()

# Check for missing or invalid values in 'Survived' column
if 'Survived' in titanic_data.columns:
    print("\nUnique values in 'Survived' column:")
    print(titanic_data['Survived'].unique())
    print("\nMissing values in 'Survived':", titanic_data['Survived'].isnull().sum())
else:
    print("\nColumn 'Survived' not found in dataset.")
    exit()

# Show survival rate vs passenger class and include actual survival counts by gender
if 'Survived' in titanic_data.columns and 'Pclass' in titanic_data.columns and 'Sex' in titanic_data.columns:
    survival_rate_by_class = titanic_data.groupby('Pclass')['Survived'].mean().reset_index()
    survival_counts_by_gender = titanic_data[titanic_data['Survived'] == 1].groupby(['Pclass', 'Sex']).size().reset_index(name='Count')

    plt.figure(figsize=(14, 8))

    # Plot survival rate by class
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(data=survival_rate_by_class, x='Pclass', y='Survived', palette='viridis', ax=ax1)
    ax1.set_title('Survival Rate by Passenger Class', fontsize=16)
    ax1.set_xlabel('Passenger Class', fontsize=12)
    ax1.set_ylabel('Survival Rate', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y')

     # Plot survival counts by gender and class
    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(data=survival_counts_by_gender, x='Pclass', y='Count', hue='Sex', palette='coolwarm', ax=ax2)
    ax2.set_title('Survival Counts by Gender and Class', fontsize=16)
    ax2.set_xlabel('Passenger Class', fontsize=12)
    ax2.set_ylabel('Survival Count', fontsize=12)
    ax2.legend(title='Gender', fontsize=10)
    
    plt.tight_layout()
    plt.show()
else:
    print("\nColumns 'Survived', 'Pclass', or 'Sex' not found in dataset.")
