import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('student-mat.csv', sep=';')
data

columns = [
    'G1',
    'G2',
    'G3',
]
model = data[columns].copy()
model

x = model.drop(['G3'], axis=1)
y = model[['G3']]
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.3, random_state=42)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

linear_coef = linear_model.coef_
linear_intercept = linear_model.intercept_
linear_predict = linear_model.predict(x_test)
linear_score = linear_model.score(x,y)
linear_mse = mean_squared_error(y_test, linear_predict)

# print(f'This is the coefficient {linear_coef}')
# print(f'This is the intercept {linear_intercept}')
# print(f'This is the prediction {linear_predict}')
# print(f'This is the score {linear_score}')
# print(f'This is the MSE {linear_mse}')

lasso_model = Lasso()
lasso_model.fit(x_train, y_train)

lasso_coef = lasso_model.coef_
lasso_intercept = lasso_model.intercept_
lasso_predict = lasso_model.predict(x_test)
lasso_score = lasso_model.score(x,y)
lasso_mse = mean_squared_error(y_test, lasso_predict)

# print(f'This is the coefficient {lasso_coef}')
# print(f'This is the intercept {lasso_intercept}')
# print(f'This is the prediction {lasso_predict}')
# print(f'This is the score {lasso_score}')
# print(f'This is the MSE {lasso_mse}')

joblib.dump(linear_model, 'linear_model.joblib')
joblib.dump(lasso_model, 'lasso_model.joblib')
# print("Model Saved")

new_student = np.array([[20, 15]])
linear_model = joblib.load('linear_model.joblib')
lasso_model = joblib.load('lasso_model.joblib')

linear_pred = linear_model.predict(new_student)
lasso_pred = lasso_model.predict(new_student)

# print("\nPredictions for new student:")
# print(f"Linear Regression prediction: {linear_pred[0]:}")
# print(f"Lasso prediction: {lasso_pred[0]:}")

plt.scatter(x_train[:,0], y_train, color='green', label='Actual')
plt.plot(x_train[:,0], lasso_predict, color='red', label='Lasso Regression Line')
plt.legend()
plt.title('Lasso Regression')
plt.xlabel('G1')
plt.ylabel('G3')
plt.grid(True)
plt.show()
