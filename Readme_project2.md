# Project 2 - Ames Housing Data and Kaggle Challenge
### Supriya Sambhus

## Problem Statement
   

As part of the Data Science team at Propnex pte ltd, a real estate company, I have been with tasked with analysing and modeling the Ames Housing Dataset and finding the key factors that influence the Sales Price of a house in Ames city. Through this analysis, we will make recommendations to homeowners on improving their property’s value, help them gaining insights about which features of their house influence its Sale Price and predict the price of their house. 

## Dataset

We were provided the Ames Housing data for individual residential properties sold in Ames, IA from 2006 to 2010. The data had two parts train.csv(2,051 rows, 81 columns) and test.csv(878 rows,80 columns). Test.csv is similar to test.csv in terms of features, except it doesn't have the dependent variable i.e. Sale Price. It will be used to predict teh Sale Price and submit to the Kaggle competition. 

## Data Cleaning

### Handling missing data
There were many missing values in the data when it was loaded into the Pandas dataframe. 
1. Values like Pool Qc, Fireplace Qu which were null becuase of no pool, fireplace etc. were update to "NA" string by checking the corresponding featires like Pool Area, Fireplaces etc.
2. Some features which were "NA" in the original file were read as null values by Python, these were updates as "NA"
3. Some features like lot_frontage and garage yr built were imputed to mean values within the same feature. 
4. For features where missing data is less than 5%, we set the values to "NA" or None.

### Handling Outliers
Extreme outliers which seem like unusual or atypical data were removed from the dataset. E.g. for Lot Area, 2 outliers greater than 100,000 squarefeet were removed. 

# EDA
### Numerical Variables
- Boxplot and histogram to assess the distribution of the variable
- Scatter plot with Sale Price to assess the correlation of the variable with Sale Price
- calculate Pearson's correlation coefficient

### Categorical Variables 
- Category wise Boxplot to assess the distribution of data in each category
- Category wise strip plot to assess the count of data in each category

# Pre-Processing
## Handling Ordinal Variables
Ordinal variables were converted to numerical by assigning a numerical value to the categories, where it was clear scale of 0 to n (representing worst to best) that could be assigned to the categories. 

## One-hot encoding
One hot encoding was applied to all the nominal categorical variables. 

## Standardisation
- StandardScaler was used to scale all the variables to bring them to a commen scale with mean = 0 and standard deviation = 1. 

# Modeling & Feature Engineering

Following steps were done to incrementally improve the model
    
### Feature Selection
    - Linear Regression
    - Lasso Regression
    - Drop unwanted, multicollinear coefficients which were reduced to zero
    - Ridge Regression
### Polynomial Fetaures
    - Lasso Regression
    - Drop unwanted, multicollinear coefficients which were reduced to zero
    - Ridge Regression
### Feature Engineering 
    - Lasso Regression
    - Drop unwanted, multicollinear coefficients which were reduced to zero
    - Ridge Regression

Models were assessed at various stages using R2 score, Cross Val score, MSE and RMSE. Please see the table showing the summary of scores. 

| Model                                                    | R2 score | Cross Val score | MSE               | RMSE       | Evaluation          |
| -------------------------------------------------------- | -------- | --------------- | ----------------- | ---------- | ------------------- |
|                                                          | Train    | Test            | Train R2- Test R2 | CrossVal   | CrossVal - Train R2 | Train | Test | Test MSE - Train MSE | Train | Test | Test RMSE - Train RMSE |  |
| Linear                                                   | 0.936    | \-7.59E+21      | 7.59E+21          | \-2.01E+22 | 2.01E+22            |    408,219,625.66 | 3.81E+33 | 4.45E+31 |      20,204.45 | 6.67E+15 | 6.67E+15 | 1\. The testing r2 score is much worse than the training score, indicating an overfit. A negative testing score shows a poor fit. <br>2\. The MSE and RMSE  values are very high, indicating a bad model. The MSE value of the testing set is much worse than the training set indicating overfitting<br>3\. A negative mean cross val score also shows that the model is very poorly fit, possibly because of non-related variables being a part of the model. |
| Lasso                                                    | 0.93     | 0.91            | 0.02              | 0.907      | 0.023               |    447,259,323.77 |    535,844,659.49 |             88,585,335.73 |    21,148.506 |     23,148.32 |                                  1,999.81 | 1\. The train and test r2 scores  are much better than for linear regression. However r2 score for testing is still a bit worsethan train score indicating possibility of overfitting.<br>2\. The MSE values much lower than the linear model, and the RMSE scores show that the predictions on the test model miss the actual score by 23,148.319$<br>3\. Cross val score is mucg closer to the train and test scores, indicating that the model generalises well on test data. |
| Ridge (after dropping 0 coefficients)                    | 0.932    | 0.905           | 0.027             | 0.911      | 0.021               |    433,920,953.72 |    571,192,062.09 |           137,271,108.38 |    20,830.769 |     23,899.63 |                                  3,068.86 | 1\. The train and test r2 scores are fairly ghood, slightly better than Lasso. However r2 score for testing is still a bit worse than train score indicating possibility of overfitting.<br>2\. The train MSE value is lower than the Lasso but test MSE is slightly higher than Lasso, and the RMSE scores show that the predictions on the test model miss the actual score by 23899.625$<br>3\. Cross val score is  closer to the train and test scores compared to Lasso, indicating that the model generalises well on test data. |
| Lasso (With Interaction Terms)                           | 0.943    | 0.924           | 0.019             | 0.924      | 0.019               |    361,755,706.53 |    418,124,303.75 |             56,368,597.23 |    19,019.877 |     20,448.09 |                                  1,428.21 | 1\. The train r2 is better than previous Lasso and Ridge models . Test r2 score is also better than previous models. The 0.02 difference in train and test scores still persists. <br>2\. The train MSE value  than the previous Lasso and Ridge models. Test MSE is also lower however difference between train and test score still presists. RMSE scores show that the predictions miss the actual score by 20448.09$<br>3\. Cross val score is  slightly closer to the train and test scores compared previous models, indicating that the model generalises better on test data. |
| Ridge (With Interaction Terms + dropping 0 coefficients) | 0.944    | 0.926           | 0.018             | 0.924      | 0.02                |    357,790,724.14 |    431,654,678.77 |             73,863,954.63 |    18,915.357 |     20,776.30 |                                  1,860.94 | 1\. The train r2 is slightly worse than the Lasso with Interaction. terms . Test r2 score is slightly better than previous models. The difference in train and test scores still persists, but is slightly better than the lasso with interaction terms. <br>2\. The train MSE value  than the previous Lasso and Ridge models. Test MSE is also lower however difference between train and test score still presists. RMSE scores show that the predictions on the test model miss the actual score by 20,776.30$<br>3\. Cross val score is  same as for Lasso with interaction, and is slightly closer to the test score compared to Lasso with interaction models, indicating that the model generalises better on test data. |
| Lasso (Polynomial Fetaures)                              | 0.967    | 0.936           | 0.031             | 0.934      | 0.033               |    210,576,089.46 |    380,304,470.51 |           169,728,381.05 |    14,511.240 |     19,501.40 |                                  4,990.16 | 1\. The train r2 score is much better than the simple lasso and ridge as well as lasso and ridge with interaction terms . The difference in train and test scores is lower than the previous models, indicating a possibility of overfitting. <br>2\. The train MSE is much better than previous models. Test MSE is also lower however difference between train and test score is much higher. RMSE scores show that the predictions on the test model miss the actual score by 19501.397$<br>3\. Cross val score is  higher than previous models, however difference betweenthe cross val score and the train score is higher than previous models. |
| Ridge (Polynomial Fetaures + dropping 0 coefficients)    | 0.973    | 0.923           | 0.05              | 0.948      | 0.025               |    171,996,335.82 |    461,147,446.83 |           289,151,111.01 |    13,114.737 |     21,474.34 |                                  8,359.61 | 1\. The train r2 score is better than the lasso with polynomial features, however the test r2 is lower . The difference in train and test scores is higher than the lasso with polynomial features, indicating a higher possibility of overfitting. <br>2\. The train MSE is higher than lasso with polynomial features. Test MSE is higher showing a bigger  difference between train and test score compared to lasso with polynomial features and previous models. RMSE scores show that the predictions on the test model miss the actual score by 21,474.344$<br>3\. Cross val score is  higher than lasso with polynomial features, and difference between the cross val score and the train score is lower than lasso with polynomial features, but higher than previous models. |




# Conclusion
After analysing the Ames dataset using Linear Regression, Lasso Regression and Ridge Regression we come to the following conclusions:

### Bigger houses fetch better Sale Price
We see that the following factors related to the house size have the some of the highest coefficients:
- High Ground living area
- High Basement Area
- High Lot Area

### Better quality houses fetch better Sale Price
We see that the following factors related to the quality of the houses also have the some of the highest coefficients:

- Overall quality
- External quality

### Good Basement quality & exposure enhances the Sale Price
We found that the interaction variable for bsmt_quality\*bsmt_exposure has a high coefficient.

### Newer houses positively influence the Sale Price
We find that Year Built has a high influence on the Sale Price, newer houses sell at higher prices. 

### Not having a garage lowers the Sale Price
Garage_Type_NA variable has a high negative coefficient showing that having no garage lowers the sale price of the house. 

# Recommendations
- Home Owners can consider undertaking renovation and improving the overall quality of the house by using material of better quality and finish. 
- They should focus on specifically improving the external quality using better quality material on the exterior.
- Home owners may set their expectations about their property value according to:
    - Size of their Ground Living Area, Basement, Lot Area
    - Year in which the house was built
    - Basement quality & exposure
    - Neighbourhood (Stone Bridge, Northridge Heights, Green Hills have better Sale Price)








