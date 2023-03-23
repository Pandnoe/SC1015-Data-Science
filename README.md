# SC1015-Data-Science

# Introduction
For SC1015 Introduction to Data Science and Artificial Intelligience mini-project, we will be using IBM HR Analytics Employee Attrition & Performance Dataset 
from Kaggle.

# Group Members:
- Ying Hao 
- Louis Ng

# Pratical Motivation
- We will soon to be working adults hence we would like to know what factors will result us leaving a particular job.

# Sample Collection
- Our dataset was taken from Kaggle IBM HR Analytics Employee Attrition & Performance.

# Problem Formulation
- What are the factors that causes people to leave their company?
- How to prevent it?

# Data Preparation
- Data Cleaning
  - We check in the dataset whether there is any null values

# Statistical Description

# Exploratory Analysis
- Coorelation
- HVPlot
  - We compare the columns against 

# Machine Learning
- Random Forest Tree Classifier: Random Forest builds multiple decision tree using randomly selected subsets of the training data and features. Once the fitting of random forest is finished, Accuracy Score and Classification Report can can be seen.

<img width="419" alt="Screenshot 2023-03-23 112745" src="https://user-images.githubusercontent.com/128039205/227099462-20b2794d-5975-4ea0-9cdd-62cdc0a38a07.png">

|Metrics| Definitions|
|--------|------------|
|`Precision`| Precision is defined as the ratio of true positives to the sum of true and false negatives|
|`Recall`| Recall is defined as the ration of true positives to the sum of true positives and false negatives.|
|`F1 Score`| The F1 is the weighted harmonic mean of precision and recall. The closer the value of F1 Score is to 1.0, the better the expected performance of the model is.|
|`Support`| Support is the number of actual occurrences of the class in the dataset. It doesn't vary between models, it just diagnoses the performation evaluation process.|

# Algorithmic Optimization
- Smote
  - In the dataset, there is an imbalance of the values, hence we use SMOTE method to deal with the over-skewed values
  
  ``` python
  oversampler= SMOTE(random_state=0)
  smote_train, smote_target = oversampler.fit_resample(train,target_train)
  ```
  
- Gradient Boosting
  - Gradient boosting is a machine learning technique that involves combining multiple weak or simple models to create a stronger model. It works by iteratively adding new models to the ensemble, with each new model focused on correcting the errors of the previous models.
  
  ``` python
  from sklearn.ensemble import GradientBoostingClassifier
  gb = GradientBoostingClassifier(**gb_params)
  gb.fit(smote_train, smote_target)
  ```
  
# Statiscal Inference

# Reference
- https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- https://www.kaggle.com/code/arthurtok/employee-attrition-via-ensemble-tree-based-methods/notebook
- https://www.kaggle.com/code/faressayah/ibm-hr-analytics-employee-attrition-performance
- https://bobbyhadz.com/blog/python-install-imbalanced-learn#install-imbalanced-learn-imblearn-in-jupyter-notebook
