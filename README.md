# SC1015-Data-Science

# Introduction
For SC1015 Introduction to Data Science and Artificial Intelligience mini-project, we will be using IBM HR Analytics Employee Attrition & Performance Dataset 
from Kaggle.

# Group Members (A134):
- Ying Hao 
- Louis Ng

# Pratical Motivation
- We are soon to be working adults therefore, we would like to know what factors are important when choosing a job that will ensure that I have a long career in that company
# Sample Collection
- Our dataset was taken from Kaggle IBM HR Analytics Employee Attrition & Performance.

# Problem Formulation
- What are the factors that causes people to leave their company?

# Data Preparation
- Data Cleaning
  - Checked if there are any missing values in the dataset and if there is remove them

<img width="250" alt="Screenshot 2023-03-30 100606" src="https://user-images.githubusercontent.com/128039205/228709564-5a7cc171-5622-46a3-8ce1-1e4633c408c4.png">

  - In our dataset, we have no null values. Hence there is no need to remove them.
  - Checked for any redundant columns in the dataset and remove them
  - From the dataset, we concluded that `EmployeeCount`,`Over18`,`EmployeeNumber`,`StandardHours`
  ``` python
  df = df.drop(['EmployeeCount', 'Over18', 'EmployeeNumber', 'StandardHours'], axis=1)
  ```

# Statistical Description

# Exploratory Analysis
- Correlation
![Correlation](https://user-images.githubusercontent.com/128039205/227194811-52cbbea8-828d-470d-8861-c1d083f31acf.png)

From the above, we can see that the many columns of the data set are poorly correlated with each other. Thus, this is a good dataset for predictive model as for predictive model, it is generally better to train a model with features that are poorly correlated with each other. 


- HVPlot
  - The reason for using HVPlot is because it is able to create good visualization of the data quickly and easily, which in our case, it is extremely useful.
  - We compare the factors (elements) against attrition factor to deduce which factor greatly affects the attrition.
 
<img width="671" alt="Screenshot 2023-03-23 193212" src="https://user-images.githubusercontent.com/128039205/227191992-996d5316-58c5-440f-97a8-10f1ad198730.png">
<img width="674" alt="Screenshot 2023-03-23 193508" src="https://user-images.githubusercontent.com/128039205/227192001-cb2a6d4b-b22a-4a24-8d91-8c5f57fb75b5.png">
<img width="671" alt="Screenshot 2023-03-23 193551" src="https://user-images.githubusercontent.com/128039205/227192007-b923513c-9734-4165-9c91-fc602d5251e7.png">
<img width="675" alt="Screenshot 2023-03-23 193616" src="https://user-images.githubusercontent.com/128039205/227192013-e14bbd99-ecb5-42ea-ac87-913ca8c4011e.png">

From the above images, we can see that those with low `JobLevel`,`MonthlyIncome`,`YearsAtCompany`, and `TotalWorkingYears` are more likely to quit there jobs.

<img width="671" alt="Screenshot 2023-03-23 194418" src="https://user-images.githubusercontent.com/128039205/227193774-847892db-efee-4439-a400-9f6b6505731a.png">

From the above image, we can see that those who is `Single` marital status (represented by 1) are most likely to quit than the `Divorced` and `Married`.

<img width="673" alt="Screenshot 2023-03-23 194216" src="https://user-images.githubusercontent.com/128039205/227194115-dd49d248-815c-4032-8ef5-e51a0342ebfa.png">

From the above image, we can see that `Male`(represented by 1) is most likely to quit the job compared to `Female` (represented by 0)


# Machine Learning
- Random Forest Tree Classifier: Random Forest builds multiple decision tree using randomly selected subsets of the training data and features. Once the fitting of random forest is finished, Accuracy Score and Classification Report can can be seen.

<img width="419" alt="Screenshot 2023-03-23 112745" src="https://user-images.githubusercontent.com/128039205/227099462-20b2794d-5975-4ea0-9cdd-62cdc0a38a07.png">

|Metrics| Definitions|
|--------|------------|
|`Precision`| Precision is defined as the ratio of true positives to the sum of true and false negatives|
|`Recall`| Recall is defined as the ration of true positives to the sum of true positives and false negatives.|
|`F1 Score`| The F1 is the weighted harmonic mean of precision and recall. The closer the value of F1 Score is to 1.0, the better the expected performance of the model is.|
|`Support`| Support is the number of actual occurrences of the class in the dataset. It doesn't vary between models, it just diagnoses the performation evaluation process.|

- Logistic Regression: A type of statistical model that estimates the probility of an event occuring based on a dataset of independent variables

We have decide to use MonthlySalary and JobLevel to conduct logistic regression on as these 2 variables have shown to have a relationship to attrition based on our extraplotary data analysis earlier.

Logistic Regression Using MonthlySalary

![image](https://user-images.githubusercontent.com/100660376/229361552-ce90c97f-ff89-4df7-9f4c-64c78a4f4714.png)

Logistic Regression Using JobLevel
![image](https://user-images.githubusercontent.com/100660376/229361567-4b7a7db7-62f2-4557-9dcc-adf9e78d9bd6.png)


The accuracy of logistic regression using MonthylySalary is only 0.497 which meant that MonthlySalary is a poor predictor of attrition according to the model. On the other hand the accuracy of logistic regression using JobLevel is 0.688 which is a decent accuracy in predicting attrition.

# Algorithmic Optimization
- Smote
  - In the dataset, there is an imbalance of the values, which result in a one-sided skewness as seen in the image below.
  
  <img width="748" alt="Screenshot 2023-03-30 090819" src="https://user-images.githubusercontent.com/128039205/228702011-d6fb36d7-b7f2-4a3c-8b24-284109d52030.png">

  Hence we use SMOTE method to deal with the over-skewed values
  
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
