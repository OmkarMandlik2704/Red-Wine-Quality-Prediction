# Red-Wine-Quality-Prediction

Project Overview

This project aims to predict the quality of red wine based on its physicochemical properties using Support Vector Machine (SVM). The dataset used for this analysis is the winequality-red.csv, which contains various attributes of red wine such as acidity, alcohol content, and pH.

Dataset Information

The dataset consists of several physicochemical properties of red wine.

The target variable is quality, which is categorized into two classes: Good and Bad.

Key Features in Dataset

Fixed acidity

Volatile acidity

Citric acid

Residual sugar

Chlorides

Free sulfur dioxide

Total sulfur dioxide

Density

pH

Sulphates

Alcohol

Quality (Target Variable)

Steps in the Project

1. Data Preprocessing

Load the dataset using Pandas.

Perform exploratory data analysis (EDA) with df.info() and df.describe().

Convert the quality column into categorical values: Good and Bad.

Perform Label Encoding for categorical values.

Handle missing values if any.

2. Exploratory Data Analysis (EDA)

Plot bar charts to analyze relationships between quality and other variables using Seaborn.

Identify patterns and trends in the data.

3. Feature Engineering

Separate features (X) and target variable (y).

Encode categorical variables using LabelEncoder.

Split the dataset into training and testing sets (80%-20% split).

4. Model Training

Train an SVM classifier using an RBF kernel.

Fit the model on the training dataset.

5. Model Evaluation

Make predictions on the test dataset.

Generate a confusion matrix and visualize it using a heatmap.

Compute accuracy score using sklearn.metrics.accuracy_score().

6. Prediction on New Data

Perform predictions on a new input sample using the trained SVM model.

Dependencies

Python

Pandas

Seaborn

Matplotlib

Scikit-learn
Results

The model achieved a certain accuracy (replace with actual accuracy from the script).

Predictions can classify wine as Good or Bad based on its properties.
Conclusion

This project successfully demonstrates how machine learning can be used to classify wine quality based on physicochemical features. The implementation of SVM provides a solid baseline model for classification tasks in the food and beverage industry
