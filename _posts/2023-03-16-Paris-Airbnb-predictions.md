---
title:  "Paris Airbnb predictions"
search: false
categories: 
  - machine learning
last_modified_at: 2023-03-16T08:06:00-05:00
---

Hey there! Welcome to our Airbnb analysis project. We hope you find the following information informative and interesting!!

Our dataset was obtained from [**Kaggle**](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews/code?datasetId=1530930) and consists of information on Airbnb accommodations across 10 major cities, including 5 million reviews spanning from November 16th, 2008, to March 1st, 2021.

In this project, our focus was on analyzing Airbnb accommodations in Paris, which comprises **37,907 listings and 972,309 reviews**. Our analysis uncovered some fascinating insights, which we are delighted to share with you.

This page is for machine learning models and predictions. For visualization, please visit [Paris Airbnb visualization](https://zixtang.github.io/visualization/Paris-Airbnb-visualization/)

---

As we delved into the Airbnb data for Paris, we stumbled upon a couple of fascinating questions that caught our attention:

1. Can we **predict the price** of an accommodation with attributes of the accommodation and host?
2. Can we predict the **popularity** of a rental by estimating **the number of reviews** it received?

To get to the bottom of these questions, we experimented with a range of machine learning models, honed in on the most effective one, and used it to make our predictions. Our process and findings are outlined below, so read on to discover what we uncovered!

<aside>
🏘️ **Table of content**

</aside>

# Data Preprocessing

To get our data ready for analysis with scikit-learn, we had to do a little bit of prep work on our categorical variables. See, scikit-learn cannot directly handle variables that have multiple levels, so we had to make some changes:

- We took the raw data set's 20 **arrondissements** and condensed them into just two categories: central (1) or non-central (0), depending on where they were located in Paris.
- Next, we turned **room type** into four separate columns (entire place, hotel room, private room, and shared room) with binary values.
- Finally, we grouped together different **amenities** and classified them into four types (essentials, stand-out, safety items, and high-demanding items) based on how often they appeared in the data set.

# Feature Selection

So, the original data set had 33 variables (or columns). But we quickly realized that some of them, like the accommodation name, host ID, were not really relevant to what we were trying to find out (namely, how price and popularity are determined). So, we gave those variables the boot after a bit of manual screening.
After that, we performed correlation analysis to the variables with price and popularity, and carefully selected the characteristics that were most strongly correlated with each outcome variable and ended up with a pretty solid set of results.

<aside>
💡 **Selected features for price prediction:**
accommodates, bedrooms, instant_bookable, host_is_superhost, central_location, amenities_essenstials, amenities_standout, amenities_safety, amenities_highdemand, entire place, hotel room, private room, shared room

</aside>

<aside>
💡 **Selected features for number of reviews prediction:**
accommodates, instant_bookable, host_is_superhost, central_location, amenities_essenstials, amenities_standout, amenities_safety, amenities_highdemand, entire place, hotel room, private room, shared room, price

</aside>

# Machine Learning Model Building

Now, here's the real meat of the matter: which machine learning model reigns as the champion when it comes to predicting price and number of reviews?
To find out, we put three of the most widely-used models to the test: **linear regression**, **polynomial regression**, and **random forest regression**. We gave them all a fair shot and carefully compared their performances to determine the ultimate winner (which we'll reveal in just a bit).
We did this process twice, once for price prediction and once for number of reviews prediction. Since the methods were pretty similar, we'll just show you the code we used for price prediction. Ready to dive in?

## Split the Dataset

We took our dataset and divided it into three separate groups: training data, cross-validation data, and test data. The training data was used to teach our models how to make predictions, while the cross-validation data was used to assess how well our models were performing. Finally, once we had selected the top-performing model, we put it to the test on the remaining test data to see just how accurate its predictions would be.

```python
# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(x_data, y_data, test_size=0.40, random_state=42)
# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=42)
# Delete temporary variables
del x_, y_
```

## Linear Regression

We first tried the linear regression model:

```python
from sklearn import preprocessing
from sklearn import linear_model

# feature scaling
scaler_linear = preprocessing.StandardScaler()
x_train_scaled= scaler_linear.fit_transform(x_train)
x_cv_scaled= scaler_linear.transform(x_cv)

# build and fit linear regression model
regr = linear_model.LinearRegression()
regr.fit(x_train_scaled, y_train)

# use the built model to predict the training data and cross-validation data
yhat_train = regr.predict(x_train_scaled)
yhat_cv = regr.predict(x_cv_scaled)

# get the MSE and  r2-score for the training data
print("Residual sum of squares (MSE) for training data: %.2f" % mean_squared_error(y_train, yhat_train))
print("R2-score for training data: %.2f" %  regr.score(x_train_scaled, y_train))
# get the MSE and  r2-score for the cross-validation data
print("Residual sum of squares (MSE) for cross-validation data: %.2f" % mean_squared_error(y_cv, yhat_cv))
print("R2-score for cross-validation data: %.2f" %  regr.score(x_cv_scaled, y_cv))
```

## Polynomial Regression

Then, we tried three polynomial regression models with degrees of 2, 3 and 4 

```python
# Initialize lists containing the results, models, and scalers
train_mses = []
cv_mses = []
models = []
scalers = []
train_scores = []
cv_scores = []

# Loop over 3 times. Each adding one more degree of polynomial higher than the last.
for degree in range(2,5):

    # Add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    
    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)
    
    # Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat)
    train_mses.append(train_mse)
    
    # Compute the training r2 score
    train_score = model.score(X_train_mapped_scaled, y_train)
    train_scores.append(train_score)
    
    # Add polynomial features and scale the cross validation set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_cv_mapped = poly.fit_transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
    
    # Compute the cross validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat)
    cv_mses.append(cv_mse)
    
    # Compute the cross validation r2 score
    cv_score = model.score(X_cv_mapped_scaled, y_cv)
    cv_scores.append(cv_score)
```

Let’s see which polynomial degree works better:

**For price prediction:**

![poly_price_mse.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bba9c54a-3ead-4471-a02a-b3d45656220a/poly_price_mse.png)

![poly_price_r2score.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/07bc04f0-7331-482f-b349-bdd852c35f58/poly_price_r2score.png)

**For number of reviews prediction:**

![poly_popularity_mse.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/defc0279-9b62-4486-8fcf-bb829c401074/poly_popularity_mse.png)

![poly_popularity_r2score.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/71872e9f-a927-4439-b8eb-1b4f45a3f4e3/poly_popularity_r2score.png)

As the y-axis scales are too large (over 1e20), it's difficult to distinguish the differences between degree = 2 and degree = 3. Therefore, we recorded the values to compare them directly in the [Model Selection](https://www.notion.so/Paris-Airbnb-predictions-d1514367a0a9439fae6d44d0013d03c9?pvs=21) part.

## Random Forest Regression

- **Step 1: Converting Continuous Variables to Binary**

Before we could get started with the scikit-learn random forest regression, we needed to make a quick adjustment. Since this method doesn't work with numerical values,  we had to find the optimal threshold for our continuous variables and then transform them into binary variables for use by the algorithm.

```python
# define a function to find the information gain
from chefboost.training import Training
config = {'algorithm': 'Regression'}

def findGain(df, column, threshold):
    idx = df[df[column] <= threshold].index
    temp_df = df.copy()
    temp_df[column] = '>'+str(threshold)
    temp_df.loc[idx,column] = '<='+str(threshold)
    gain = Training.findGains(temp_df,config)['gains'][column]
    return threshold, gain
```

```python
# find thresholds of continuous predictors for price prediction
accommodates_binary = data[['accommodates', 'price']]
accommodates_binary.rename(columns={"price": "Decision"}, inplace = True)
accommodates_unique_values = sorted(data['accommodates'].unique())
accommodates_gains = []
for value in accommodates_unique_values:
    accommodates_gains.append(findGain(accommodates_binary, 'accommodates', value))
sorted_accommodates = sorted(accommodates_gains, key=lambda tup: tup[1])
threshold_accommodates = sorted_accommodates[-1]

bedrooms_binary = data[['bedrooms', 'price']]
bedrooms_binary.rename(columns={"price": "Decision"}, inplace = True)
bedrooms_unique_values = sorted(data['bedrooms'].unique())
bedrooms_gains = []
for value in bedrooms_unique_values:
    bedrooms_gains.append(findGain(bedrooms_binary, 'bedrooms', value))
sorted_bedrooms = sorted(bedrooms_gains, key=lambda tup: tup[1])
threshold_bedrooms = sorted_bedrooms[-1]

print(threshold_accommodates)
print(threshold_bedrooms)
```

With these steps, we successfully determined the optimal thresholds for `accommodates` and `bedrooms`, enabling us to convert them into binary variables.

```python
data['accommodates_morethan5'] = np.where(data['accommodates'] > 5, 1, 0)
data['bedrooms_morethan3'] = np.where(data['bedrooms'] > 3, 1, 0)
```

A similar approach was used to convert continuous variables to binary for number of reviews prediction using random forest regression.

- **Step 2: Find the Best Parameters for Random Forest Regression**

We then used `GridSearchCV` to find the optimal parameters to be used in random forest regression, and the optimal parameters are `max_depth = 32`, and `min_samples_split = 100`.

```python
# search for the best parameters for random forest regression
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth': [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, None], 
               'min_samples_split': [100, 400, 1200, 2000, 4000, 8000, 12000, 28000]}]
RFR = RandomForestRegressor()
Grid_result = GridSearchCV(RFR, parameters, cv = 4)
Grid_result.fit(x_train, y_train)
BestRFR = Grid_result.best_estimator_
```

- **Step 3: Establish and Fit Random Forest Regression**

```python
RFR = RandomForestRegressor(n_estimators=100, criterion = "mse", max_depth=32, min_samples_split=100)
RFR.fit(x_train, y_train)

# evaluation
score_train = RFR.score(x_train, y_train)
score_cv = RFR.score(x_cv, y_cv)

yhat_train = RFR.predict(x_train)
train_mse = mean_squared_error(y_train, yhat_train)
yhat_cv = RFR.predict(x_cv)
cv_mse = mean_squared_error(y_cv, yhat_cv)

#Compute the training MSE and R2 score
print("Training mse is:", train_mse)
print("Training score is:", score_train)
#Compute the cross validation MSE and R2 score
print("CV mse is:", cv_mse)
print("CV score is:", score_cv)
```

# Model selection

Now let’s take all models together and have a look on their performance.

For price prediction models:

|  | Training - MSE | Training - R2 | CV - MSE | CV - R2 |
| --- | --- | --- | --- | --- |
| Linear regression | 30973.83 | 0.12 | 45885.77 | 0.08 |
| Polynomial regression (degree = 2) | 29598.55 | 0.16 | 44672.02 | 0.11 |
| Polynomial regression (degree = 3) | 28461.11 | 0.19 | **44057.44** |  **0.12** |
| Polynomial regression (degree = 4) | 25863.11 | 0.27 | 4.38e+25 | -8.75e+20 |
| Random forest regression | 28186.23 | 0.20 | 44924.21 | 0.10 |

<aside>
💡 **Conclusion:**
**The winner model for price prediction is: polynomial regression with degree of 3!**

</aside>

For number of reviews prediction models:

|  | Training - MSE | Training - R2 | CV - MSE | CV - R2 |
| --- | --- | --- | --- | --- |
| Linear regression | 1924.76 | 0.10 | 1818.50 | 0.11 |
| Polynomial regression (degree = 2) | 1864.29 | 0.13 | 1778.09 | 0.13 |
| Polynomial regression (degree = 3) | 1802.10 | 0.16 | 2889.94 |  -0.42 |
| Polynomial regression (degree = 4) | 1680.90 | 0.22 | 2.84e+25 |  -1.39e+22 |
| Random forest regression | 1815.32 | 0.15 | **1776.79** | **0.13** |

<aside>
💡 **Conclusion: 
The winner model for number of reviews prediction is: random forest regression!**

</aside>

# Test data prediction

Now let’s use the winner models for prediction!

## Price prediction

```python
poly = PolynomialFeatures(3, include_bias=False)
X_test_mapped = poly.fit_transform(x_test)

# Scale the test set
X_test_mapped_scaled = scalers[3-1].transform(X_test_mapped)

# Compute the test MSE
yhat = models[3-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat)
test_score = models[3-1].score(X_test_mapped_scaled, y_test)

print(f"Test MSE: {test_mse:.2f}")
print(f"Test R squared: {test_score:.4f}")
```

<aside>
💡 MSE for test data prediction is 52713.14.
R-squared score for test data prediction is 0.10.

</aside>

**The distribution of the actual and predicted price:**

![price_prediction_distribution.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/967cd1b2-505f-4d8b-8117-c67482a54fe1/price_prediction_distribution.png)

## Number of reviews prediction

```python
# use ranfom forest regression
RFG = RandomForestRegressor(n_estimators=100, criterion = "mse", max_depth=8, min_samples_split=100)
RFG.fit(x_train, y_train)

# evaluation
score_test = RFG.score(x_test, y_test)

yhat_test = RFG.predict(x_test)
test_mse = mean_squared_error(y_test, yhat_test)

print("test score is:", score_test)
print("test mse is:", test_mse)
```

<aside>
💡 MSE for test data prediction is 1588.07.
R-squared score for test data prediction is 0.12.

</aside>

**The distribution of the actual and predicted number of reviews:**

![Popularity_prediction_distribution.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c1c0cc16-b33f-448f-9576-dba138014228/Popularity_prediction_distribution.png)

<aside>
💡 **There was a significant overlap between the predicted values and the real values for both price and number of reviews!**

</aside>

---

Thank you for taking the time to read through our analysis of the Airbnb dataset in Paris. We hope that our findings have provided you with some insights into the Airbnb market in the city. We appreciate your interest in our work, and if you have any questions or feedback, please don't hesitate to contact us (📧 [Zixuan Tang](mailto:zixuantang.suki@gmail.com), 📧 [Ke Chen](mailto:kechen.kc94@gmail.com))!

Have a great day! 

> Python package used: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `chefboost`
>