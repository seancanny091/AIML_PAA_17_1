# AIML_PAA_17_1 - Comparing Classifiers

In this practical application, the goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone. The dataset used in the analysis is available here:

[Link to Banking Dataset](/data/bank-additional-full.csv) 
[Link to Banking Data Descriptive Information](/data/bank-additional-names.txt)

The programming language used was Python, and the libraries used were: pandas, seaborn, matplotlib, and sklearn.
The specifics of the analysis, including code, visualizations, comments, and observations are contained in the following Jupiter Notebook:

[Link to Jupyter Notebook](/PAA_17_1.jpynb)

## Business Understanding

The primary business objective of this task is to develop a predictive model that can accurately identify whether a client will subscribe to a term deposit based on various demographic, socio-economic, and marketing-related features. This model will help the bank to:

- Improve Marketing Campaigns
- Optimize Resource Allocation
- Enhance Customer Relationship Management
- Increase Conversion Rates

## Data Understanding

The dataset contains the following features:

# bank client data
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

The dataset collected is related to 17 campaigns that occurred between May 2008 and November 2010, corresponding to a total of 79354 contacts. There are no missing values in the dataset. Relative to the size of the dataset very few duplicates found - should have minimal impacts on the models.

## Data Preparation

The following initial data preparation steps were performed:
* Remove the 'duration' feature as it is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should be discarded if the intention is to have a realistic predictive model.

### Categorical Features

Histogram Visualizations:
![Image](/images/CATHist.png) 

Categorical features were processed to handle missing values by replacing them with the most frequent value in each column. We will also use OneHotEncoder to convert categorical variables into a format that can be provided to machine learning algorithms to do a better job in prediction. The OneHotEncoder is set to ignore any unknown categories during the transformation (handle_unknown='ignore'). This preprocessing pipeline ensures that categorical data is appropriately handled and encoded for further analysis or modeling.

### Numerical Features

Histogram Visualizations:
![Image](/images/NUMHist.png) 

Boxplot Visualizations & Stats - pre-outlier handling:
![Image](/images/NUMBoxPreOut.png) 
![Image](/images/PreOutlier.png)

Looking at plots of numerical/continuous features, we can see that there are outliers in the data.  These outliers were processed using the IQR method.

Boxplot Visualizations & Stats - post-outlier handling:
![Image](/images/NUMBoxPreOut.png) 
![Image](/images/PostOutlier.png) 

In addtion to handling outliers, numerical features were processed to handle missing values by replacing them with the median value of each column. This is useful for numerical data to maintain robustness against outliers. StandardScaler was used to standardize the features by removing the mean and scaling to unit variance. This preprocessing pipeline ensures that numerical data is cleaned and scaled properly.

#### Target Variable Visualization

![Image](/images/TargVar.png) 

The target variable is heavily imbalanced. This may negatively impact the performance of the models.


### Train/Test Data Split

The prepared dataset was split into training and testing dataset per following code:

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`

### Baseline Model

To establish a baseline performance, we can use the most frequent class in the target variable as our baseline classifier. The idea is to predict the most frequent class for all instances and measure the performance of this simple strategy.

**Baseline Accuracy: 0.8865015780529255**

### Simple Logistic Regression Model

A simple logistic regression model was build with the following parameters:

`log_reg = LogisticRegression(max_iter=1000, random_state=42)`










