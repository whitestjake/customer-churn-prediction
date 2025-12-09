Section 1: Background and Introduction

Customer churn prediction is both a useful and vital application of classification modeling that can be used in banking, subscription-based businesses and with some adjustment is applicable to retail as well. Being able to determine which customers have a high risk of leaving or canceling their service accurately is important for reducing potential loss of revenue by directly targeting these customers to use any retention strategies that might preserve their interest. For this project I developed three different machine learning models to predict customer churn using the “Bank Customer Churn Prediction” dataset sourced from Kaggle (https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset/data). This dataset contains 10,000 different unique customer data with information such as location, demographics, credit information, account activity and whether or not they stayed or exited after being made an offer (target value).
The goal of this project is to build, compare and interpret classification models that are able to predict whether or not a customer will leave. Once that determination was made, it is then possible to determine which features have the strongest impact on whether or not a customer denies service as well. The process this project takes is to first examine some of the data attributes and determine what level of data pre-processing is needed. The selection of different models and then any hyperparameter tuning and evaluation using both validation and test datasets.


Section 2: Methods

Data Preparation
The columns “RowNumber”, “CustomerId” and “Surname” were removed because they did not add any predictive value. Since there were no missing or null values to deal with in this dataset the only necessary preprocessing was one hot encoding the categories “Geography” and “Gender” while all numerical variables were scaled using z-score.
Train-Validation-Test Split
The dataset was divided into three sets for model evaluation
	70% Train
	10% Validation
	20% Test
Evaluation Models
Three supervised classification models were implemented using scikit-learn and XGBoost
1.	Logistic Regression
a.	Baseline model
2.	Random Forest Classifier
a.	More complex model for better prediction
3.	XGBoost
a.	Second complex model for better prediction
Evaluation Metrics
Evaluated models using multiple scoring methods which were accuracy, precision, recall, f1-score and ROC-AUC. Using multiple metrics allowed for the ability to identify the minority churn class (lowest frequency examples) and measure the models performance



Section 3: Experiment, Results and Discussion

Model Performance

The XGBoost model had the strongest validation performance and outperformed other models in ROC-AUC, recall and F1-score. The baseline model did not perform well at predicting churning but did perform well at predicting that they wouldn’t.
Table 1. Validation Performance Summary
Model	Accuracy 	Precision (0,1)	Recall (0,1)	F1 (0,1)	ROC-AUC 
Logistic
Regression	0.82	(0.84, 0.58)	(0.95, 0.25)	(0.89, 0.35)	0.76
Random
Forest	0.87	(0.88, 0.81)	(0.97, 0.46)	(0.92, 0.59)	0.86
XGBoost	0.87	(0.88, 0.76)	(0.96, 0.48)	(0.92, 0.59)	0.86

Test Performance

After determining XGBoost to be the best model overall, it was then evaluated on the untouched testing dataset and the results were consistent with the performance of the model on the validation set which demonstrates minimal overfitting of the data. The strong ROC-AUC score shows that the model effectively ranks customers by churn risk which is more usable than just accuracy by itself as we want to be able to put more focus where it is necessary and less where it is not when it comes to preventing customers churning on the operations side.
Discussion
Analysis of the importance of different features revealed that age, account balance, whether or not they lived in Germany and number of products being purchased were the strongest indicators for whether or not a customer will churn.  The model also demonstrated that customers with fewer products were more likely to leave than customers with multiple products (in banking products would be different types of accounts such as checking, savings, credit cards, different loans) which suggests that the more products someone has the more loyal they have to a brand. The bank could use these insights to develop strategies for retention that target at risk individuals.


Figure 1. ROC Curve for XGBoost
<img width="922" height="656" alt="image" src="https://github.com/user-attachments/assets/f8f5feea-ef21-4cfb-ae36-f3bcd26062c9" />

 
Figure 2. Feature Importance for XGBoost
<img width="923" height="551" alt="image" src="https://github.com/user-attachments/assets/fe5c00ee-12ef-4a30-b559-38a1140611f0" />

 
Section 4: Conclusion

Using different machine models this project was able to demonstrate the applicability of machine learning in predicting the likelihood of a customer churning using real banking data. After some data preprocessing and model comparison, the XGBoost classifier demonstrated the best predictive performance on both the validation and test sets. It also determined the key features such as age, geography and product usage had the highest impact on customer churn behavior. Using the results found from this classification model the bank could reduce churn by focusing on customers who are identified as the most high-risk by the model. To improve model effectiveness, it is possible to add more features that represent other behaviors or do more parameter tuning to increase the focus on already known high impact features.

Bibliography Section
1.	Kaggle Dataset: Bank Customer Churn Prediction
a.	URL: (https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset/data)
2.	Scikit-learn documentation
a.	URL: (https://scikit-learn.org/stable/)
3.	XGBoost Documentation
a.	URL: (https://xgboost.readthedocs.io/en/stable/)
4.	In class lessons on Logistic Regression and Random Forest/Ensemble modeling
