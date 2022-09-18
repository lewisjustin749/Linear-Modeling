# Linear-Modeling
 Basic exploratory data analysis, feature engineering, and then implement and perform various types of regression algorithms on e-commerce data

**Introduction**

In this assignment, you have to perform the required basic exploratory data analysis, feature engineering, and then implement and perform various types of regression algorithms we have learned so far. For each section, you must have to put the question and an appropriate header text as a text cell. And after the header, you can use multiple cells for coding and explaining, and plotting.

**Dataset**

ecommarce.csv 

**Context**

An e-commerce company based in Florida sells clothing Online. However, they also have style sessions and clothing advice sessions at their on-site locations. In an in-store session, customers come into the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want. 

The company is trying to decide whether to focus its efforts on its mobile app experience or its website. Also, they would like to build a predictive model to get an idea of the yearly amount spent by a customer based on their data. They've hired you on contract to help them figure it out! 

Additionally, your task is not only to solve the problem but also to show that you can apply and understand various types of regression techniques we have learned in the class

**Data Description:**

The data set has a list of features and most of the data are artificially generated. So, don't think these email addresses, home addresses, and credit cards are real data!

**List of Features**

Email: Unique email address of the customer
Address: Address of the customer
Credit Card: Credit cared of the customer 
Avg. Session Length: Average session of in-store style advice sessions.
Time on App: Average time spent on App in minutes
Time on Website: Average time spent on Website in minutes
Length of Membership: How many years the customer has been a member.

**Tasks**

**1. Load Data and perform general EDA [14 pts]**

* import libraries: pandas, numpy, matplotlib (set %matplotlib inline), matplotlib’s pyplot, seaborn, missingno, scipy’s stats, sklearn (1 pt)
* import the data to a dataframe and show the count of rows and columns (1 pt)
* Show the top 5 and last 5 rows (1 pt)
* call the describe method of dataframe to see some summary statistics of the numerical columns. (1 pt)
* Explain in words about the description of any two variables (1 pt)
 * Show any missing value analysis  (1 pt)
 * Plot various scatter plots to understand the data: (8 pts)
   * Yearly amount Spent vs Time on Website
   * Yearly amount Spent vs Time on App
   * Length of membership vs Time on App
   * Generate sns pairplot. Based on the plots, what feature is mostly correlated with the yearly amount spent?
   * Also, plot sns heatmsp based on correlation with annot=True and discuss which columns must be removed based on that and which column is mostly interesting and     related to Yearly Amount Spent?
   * Generate a scatter plot with the interesting column you found in the last step against the Yearly Amount Spent

**2.Feature Selection and Pre-processing [ 4 pts]**
Based on the EDA and null analysis, drop the unnecessary columns for the regression. (Don't remove Time on Website as it is part of our analysis)

**3.X/Y and Training/Test Split [8 pts]**
* Use sklearn's train_test_split to split the data set into training and test sets. There should be 30% records in the test set. The random_stat should be 101 
* As we will be doing gradient descent as well as some other regression technique, scaling the data set is important. So, use sklearn's StandardScaler for scalling the X of training and test sets. But don't do it for y(target) train and test. [For help, you can see the answer for this question: Link (Links to an external site.)

**4.Training Linear Model using SKLearn's LinearRegression  [ 25 pts]**
* Train a linear model using Sklearn''s LinearRegression (example in the linear regression slide/colab links in webcourses)
* After training, show the coefficients and intercept
* Predict for the test data
* Generate a scatter plot that shows the Y test on x-axis and y predicted in y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)
* Interpretation: Interpret the coefficient and which coefficient belongs to which feature and based on that explain any strategy that should help the business

**5.Normal Equation [25 pts]**(while solving this, you might need to convert your dataframe into various different data structures such as to_numpy(), might need to reshape, perform Transpose, add x0 columns, etc. I would recommend you to see the colab link I have shown you in the class and try to compare the shape of x and y and their data to get an idea during this process. Also, the code and discussion in the slide will help.
* Implement Normal Equation and find best_theta values based on the training set
* Display the theta values. Are they very close to the sklearn's linear regression?
* Prepare the test set before prediction
* Perform prediction for the test set
* Generate a scatter plot that shows the Y test on x-axis and y predicted in y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)
* What is the limitation of using the Normal equation for regression?

**6.Batch Gradient Descent [25 pts]**
* Implement Batch Gradient Descent based on the way we have learned in the class (See sample code form pdf). You can play with eta and n_iterations and should set to reasonable eta and number of iterations so that you can get the thetas close to Normal equation's theta
* Display the theta values. Are they very close to the sklearn's linear regression?
* Also plot step number (in x-axis) against the cost(y axis). See an example from this colab link : Link (Links to an external site.)
* Perform Prediction for the test set
* Generate a scatter plot that shows the Y test on the x-axis and y predicted in the y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)
* Short Question: How do derivatives help in the process of gradient descent?
* Short Question: What are the benefits and the limitations of using batch gradient descent? 

**7.Stochastic Gradient Descent [ 25 pts]**
* Implement Stochastic Gradient Descent and train our data set. You must have to use learning_schedule (see example code in pdf as well as the colab link I have shared in #6 above. The parameters should be reasonable and the theta values should be very close to the normal equation
* Display the theta values. Are they very close to the sklearn's linear regression?
* Also plot step number (in x-axis) against cost(y-axis). See an example from this colab link : Link (Links to an external site.)
* Perform Prediction for the test set
* Generate a scatter plot that shows the Y test on the x-axis and y predicted in the y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)
* Short Question: What are the benefits and the limitations of using Stochastic gradient descent?

**8.SGDRegressor from sklearn [25 pts]**
* Use sklearn's SGDRegressor to train a model for our data set. Put a reasonable iteration and tolerance and learning steps so that we can get coefficients close to normal equation
* Display the theta values. Are they very close to sklearn's linear regression?
* Predict for the test data
* Generate a scatter plot that shows the Y test on the x-axis and y predicted in the y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)

**9.Mini-batch Gradient Descent [ 3 pts]**
* Briefly explain how mini-batch can overcome the limitations of Batch gradient descent and SGD.

**10.Polynomial of degree 2 [ 10 pts]**
* Use sklearn's Polynomial features to degree = 2 on our training and test set
* Use linearRegression on the new polynomial features
* Predict for test set
* Generate a scatter plot that shows the Y test on the x-axis and y predicted in the y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)

**11.Polynomial of degree 3 [5 pts]**
* Use sklearn's Polynomial features to degree = 3 on our training and test set
* Use linearRegression on the new polynomial features
* Predict for test set
* Generate a scatter plot that shows the Y test on the x-axis and y predicted in the y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)

**12.Learning Curve [ 7 pts]**
* Generate learning curve with linearRegression
* Generate learning curve with polynomial regression with degree  = 5
* Interpret the result

**13.Regularization [ 3 pts]**
* Explain the purpose of regularization
* For the following Regularization methods (number 14, 15 16, 17), use the polynomial degree 3 data set

**14.Ridge Regression [ 8 pts]**
* Use sklearn's Ridge to train the data set (use the polynomial degree 3 data set)
* Predict for test set
* Generate a scatter plot that shows the Y test on the x-axis and y predicted in the y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)

**15.SGDRegressor  for Ridge [ 8 pts]**
* Use sklearn's SGDRegressor for Ridge Regression
* Predict for test set
* Generate a scatter plot that shows the Y test on the x-axis and y predicted in the y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)

**16.Lasso Regression [ 10 pts]**
* Use sklearn's Lasso
* Predict for test set
* Generate a scatter plot that shows the Y test on the x-axis and y predicted in the y-axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE, and R^2  (see documentation of sklearn's metrics)
* How Lasso perform the regularization and how does that affect the thetas?

**17.Elastic Net [10 pts]**
* Use sklearn's ElasticNet
* Predict for test set
* Generate a scatter plot that shows the Y test in x axis and y predicted in y axis
* Use sklearn's metrics to print the value of MAE, MSE, RMSE and R^2  (see documentation of sklearn's metrics)
* How ElasticNet different compared to Lasso and RIDGE perform the regularization and how does that affect the thetas

**Bonus Question: [ 4 pts]**
* In most of the above cases, for example, LinearRegression of sklearn, (Q4 above), we have used scaled data set for training. However, in a real-life scenario, you would like to predict the yearly amount spent for a new instance. 
* The real data will not be scaled. How would you use the model for this case to predict this instance? 
* [35.49726772511229,12.655651149166752,39.57766801952616,4.082620632952961] = ? 
* Write necessary code so that it will predict a reasonable value for the amount spent. 
* This is very close to our first training record.
