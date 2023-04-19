# %% [markdown]
# # Logistic Regression

# %% [markdown]
# Logistic Regression is a Machine Learning algorithm which is used for the classification problems, it is a predictive analysis algorithm and based on the concept of probability. The logistic function (also known as the sigmoid function) is used to transform the output of the linear model into a probability value between 0 and 1. The model is then fit to the training data by estimating the coefficients of the predictor variables that maximize the likelihood of the observed data. The model is then used to predict the probability of the target variable for new data.

# %% [markdown]
# # Sigmoid Function

# %% [markdown]
# In order to map predicted values to probabilities, we use the Sigmoid function. The function maps any real value into another value between 0 and 1.
# 
# $$\sigma(x) = \frac{1}{1+e^{-x}}$$
# 
# <figure style="text-align:center;">
#     <img src='https://miro.medium.com/v2/resize:fit:1400/1*JHWL_71qml0kP_Imyx4zBg.png' alt='sigmoid'/>
# </figure>
# 
# The logistic function maps any input value to a value between 0 and 1, which can be interpreted as a probability. In logistic regression, the input to the logistic function is a linear combination of the predictor variables, which is given by:
# 
# $$\hat{y} = \sigma(\beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n)$$
# 
# where $\hat{y}$ is the predicted probability of the target variable, $\beta_0$ is the intercept, $\beta_1$ is the coefficient of the first predictor variable, $\beta_2$ is the coefficient of the second predictor variable, and so on. The logistic function is shown below.

# %%
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
# Load the iris dataset
iris = load_iris()

# %%
dir(iris)
iris.feature_names

# %%
# Inspect the dataset
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
sns.pairplot(df,hue='target')
plt.show()

# %%
# Load features and target
X = df[iris.feature_names].values
y = df['target'].values

# %%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Create a logistic regression model and fit it to the training data
# Use the solver 'liblinear'
logReg = LogisticRegression(solver='liblinear', random_state=42)
logReg.fit(X_train, y_train)

# %%
# Use the model to predict the test data labels
y_pred = logReg.predict(X_test)
# print(y_pred)
# print(y_test)

# %%
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
#accuracy



