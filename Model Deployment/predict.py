import json
import pickle
import pandas as pd
from flask import jsonify

def predict_response(input_features):
    try:
        with open("../Artifacts/training_features.json", "r") as file:
            features = json.load(file)['features']
        model = pickle.load(open('model.pkl', 'rb'))
        scaling = pickle.load(open('scaling.pkl', 'rb'))
        feature_columns = [
            'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
            'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
            'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
            'YearsWithCurrManager', 'BusinessTravel_Non-Travel',
            'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
            'Department_Human Resources', 'Department_Research & Development',
            'Department_Sales', 'Education_Bachelors', 'Education_Intermediate',
            'Education_Masters', 'Education_PhD', 'Education_Schooling',
            'EducationField_Human Resources', 'EducationField_Life Sciences',
            'EducationField_Marketing', 'EducationField_Medical',
            'EducationField_Other', 'EducationField_Technical Degree',
            'EnvironmentSatisfaction_High', 'EnvironmentSatisfaction_Low',
            'EnvironmentSatisfaction_Medium', 'EnvironmentSatisfaction_Very high',
            'Gender_Female', 'Gender_Male', 'JobInvolvement_High',
            'JobInvolvement_Low', 'JobInvolvement_Medium',
            'JobInvolvement_Very high', 'JobLevel_Fresher', 'JobLevel_Junior',
            'JobLevel_Manager', 'JobLevel_Senior', 'JobLevel_Team Lead',
            'JobRole_Healthcare Representative', 'JobRole_Human Resources',
            'JobRole_Laboratory Technician', 'JobRole_Manager',
            'JobRole_Manufacturing Director', 'JobRole_Research Director',
            'JobRole_Research Scientist', 'JobRole_Sales Executive',
            'JobRole_Sales Representative', 'JobSatisfaction_High',
            'JobSatisfaction_Low', 'JobSatisfaction_Medium',
            'JobSatisfaction_Very high', 'MaritalStatus_Divorced',
            'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_No',
            'OverTime_Yes', 'PerformanceRating_Excellent',
            'PerformanceRating_Outstanding', 'RelationshipSatisfaction_Excellent',
            'RelationshipSatisfaction_Good', 'RelationshipSatisfaction_Outstanding',
            'RelationshipSatisfaction_Poor', 'WorkLifeBalance_Best',
            'WorkLifeBalance_Better', 'WorkLifeBalance_Good',
            'WorkLifeBalance_Poor'
        ]

        # categorical columns and their categories
        categorical_columns = {
            'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
            'Department': ['Human Resources', 'Research & Development', 'Sales'],
            'Education': ['Bachelors', 'Intermediate', 'Masters', 'PhD', 'Schooling'],
            'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
            'EnvironmentSatisfaction': ['High', 'Low', 'Medium', 'Very high'],
            'Gender': ['Female', 'Male'],
            'JobInvolvement': ['High', 'Low', 'Medium', 'Very high'],
            'JobLevel': ['Fresher', 'Junior', 'Manager', 'Senior', 'Team Lead'],
            'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director',
                        'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'],
            'JobSatisfaction': ['High', 'Low', 'Medium', 'Very high'],
            'MaritalStatus': ['Divorced', 'Married', 'Single'],
            'OverTime': ['No', 'Yes'],
            'PerformanceRating': ['Excellent', 'Outstanding'],
            'RelationshipSatisfaction': ['Excellent', 'Good', 'Outstanding', 'Poor'],
            'WorkLifeBalance': ['Best', 'Better', 'Good', 'Poor']
        }

        processed_data = pd.DataFrame(columns=feature_columns)

        # for categorical columns
        for column, categories in categorical_columns.items():
            for category in categories:
                processed_data[f'{column}_{category}'] = [1 if input_features.get(column) == category else 0]

        # for numerical columns
        numerical_columns = [
            'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        for column in numerical_columns:
            processed_data[column] = [input_features.get(column, 0)]

        # order
        processed_data = processed_data.reindex(columns=feature_columns, fill_value=0)

        # Scaling
        scaled_data = scaling.transform(processed_data)

        # Predict using the model
        prediction = model.predict(scaled_data)[0]

        # Determine the result
        result = "The employee is expected to remain with the company." if prediction == 0 else "The chances are high that the employee will leave the company."

        # Return the prediction in the response
        return result
    except Exception as e:
        print("Prediction error:", str(e))
        raise
