# What This Code Does:

-This project is a full end-to-end machine learning workflow. Here’s what happens step by step:

## 1-Data Loading & Inspection:

  -Loads the Telco Customer ***Churn dataset*** from a URL.

  -Displays the first 40 rows and overall dataset shape.

  -Checks for ***missing values*** and inspects data types.

## 2-Data Cleaning & Preprocessing:

  -Converts ***TotalCharges*** to numeric and handles any invalid entries.

  -Drops rows with ***missing values and irrelevant columns*** like customerID.

  -***Encodes*** categorical features using ***LabelEncoder*** so models can process them.

## 3-Train/Test Split:

  -Splits the dataset into ***training*** and ***testing*** sets (80/20).

## 4-Model Setup & Pipelines:

  -Prepares three models:

  >***Support Vector Machine (SVM)***

  >***K-Nearest Neighbors (KNN)***

  >***Random Forest***

  -SVM and KNN models are wrapped in ***pipelines*** with ***standard scaling*** for better performance.

## 5-Cross-Validation:

  -Evaluates each model using ***5-fold stratified cross-validation*** on the ***training data***.

  -Reports ***mean accuracy*** and ***standard deviation*** to assess model stability.

## 6-Test Set Evaluation

  -***Trains*** each model on the training data.

  -Makes ***predictions*** on the test set and calculates ***accuracy***.

  -Displays a ***full classification report*** (precision, recall, F1-score).

## 7-Model Comparison Visualization

  -Creates a bar ***chart*** showing the test set ***accuracy*** of all models.

  -Saves the chart as model_comparison.png for easy reference.


# Results:

-At the end of the workflow, you can quickly see ***which model performs best on unseen data***. This is especially useful for companies who want to focus on the ***right model to predict churn and take preventive action to retain customers***.
