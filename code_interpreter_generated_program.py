# Required Libraries
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_pdf import PdfPages

# Data Cleaning
def clean_data(df):
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Fill missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    
    return df

# Descriptive Statistics
def generate_descriptive_statistics(df):
    stats = {}
    female_approved = df[(df['Gender'] == 'Female') & (df['Loan_Status'] == 'Y')]
    total_female = df[df['Gender'] == 'Female']
    stats['Percentage of female applicants that had their loan approved'] = (len(female_approved) / len(total_female)) * 100 if len(total_female) != 0 else 0
    
    stats['Average income of all applicants'] = df['ApplicantIncome'].mean()
    stats['Average income of all applicants that are self-employed'] = df[df['Self_Employed'] == 'Yes']['ApplicantIncome'].mean()
    stats['Average income of all applicants that are not self-employed'] = df[df['Self_Employed'] == 'No']['ApplicantIncome'].mean()
    stats['Average income of all graduate applicants'] = df[df['Graduate'] == 1]['ApplicantIncome'].mean()
    
    graduate_approved = df[(df['Graduate'] == 1) & (df['Loan_Status'] == 'Y')]
    total_graduates = df[df['Graduate'] == 1]
    stats['Percentage of graduate applicants that had their loan status approved'] = (len(graduate_approved) / len(total_graduates)) * 100 if len(total_graduates) != 0 else 0
    
    return stats

# Save Descriptive Statistics to File
def save_statistics_to_file(stats, filename):
    with open(filename, 'w') as file:
        for key, value in stats.items():
            file.write(f"{key}: {value}\n")

# Exploratory Data Analysis & Visualizations
def data_exploration(df, filename):
    with PdfPages(filename) as pdf:
        # Distribution of loan approval status
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Loan_Status', data=df, palette='pastel')
        plt.title('Distribution of Loan Approval Status')
        pdf.savefig()
        plt.close()
        
        # Distribution of male vs. female applicants
        plt.figure(figsize=(8, 6))
        gender_dist = df['Gender'].value_counts()
        gender_labels = ['Male', 'Female']
        plt.pie(gender_dist, labels=gender_labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel", 2))
        plt.title('Distribution of Male vs. Female Applicants')
        pdf.savefig()
        plt.close()
        
        # Distribution of applicant incomes
        plt.figure(figsize=(8, 6))
        sns.histplot(df['ApplicantIncome'], bins=50, color='skyblue', kde=True)
        plt.title('Distribution of Applicant Incomes')
        pdf.savefig()
        plt.close()
        
        # Average income based on education
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Education', y='ApplicantIncome', data=df, palette='pastel', ci=None)
        plt.title('Average Income Based on Education')
        pdf.savefig()
        plt.close()
        
        # Distribution based on property area
        plt.figure(figsize=(8, 6))
        property_area_dist = df['Property_Area'].value_counts()
        property_area_labels = ['Urban', 'Semi-Urban', 'Rural']
        plt.pie(property_area_dist, labels=property_area_labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel", 3))
        plt.title('Distribution of Applicants Based on Property Area')
        pdf.savefig()
        plt.close()


# Model Generation & Training
def train_model(df):
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Married', 'Self_Employed', 'Property_Area'], drop_first=True)
    
    # Split data
    X = df_encoded.drop('Loan_Status', axis=1)
    y = df_encoded['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    
    return logreg, scaler

# Predictive Analysis
def predict_sample(logreg, scaler, sample_data):
    sample_data_scaled = scaler.transform(sample_data)
    prediction = logreg.predict(sample_data_scaled)
    return "Approved" if prediction[0] == 1 else "Not Approved"

# Main Execution
if __name__ == "__main__":
    # Load Data
    loan_data_path = "loan_data.xlsx"
    loan_data = pd.read_excel(loan_data_path)
    loan_data.set_index('Loan_ID', inplace=True)

    # Clean Data
    cleaned_loan_data = clean_data(loan_data.copy())

    # Generate and Save Descriptive Statistics
    statistics = generate_descriptive_statistics(cleaned_loan_data)
    statistics_file_path = "descriptive_statistics.txt"
    save_statistics_to_file(statistics, statistics_file_path)

    # Generate and Save Exploratory Data Analysis & Visualizations
    visualization_file_path = "visualizations.pdf"
    data_exploration(cleaned_loan_data, visualization_file_path)

    # Train Model
    logreg_model, data_scaler = train_model(cleaned_loan_data)

    # Sample Predictive Analysis Session
    sample_data = cleaned_loan_data.iloc[0].drop('Loan_Status').values.reshape(1, -1)  # Using first row as a sample
    prediction_result = predict_sample(logreg_model, data_scaler, sample_data)
    print(prediction_result)

