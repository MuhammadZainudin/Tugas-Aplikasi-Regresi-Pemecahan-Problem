import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('Student_Performance.csv')

X = df[['Hours Studied']].values
y = df['Performance Index'].values

linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

X_squared = X ** 2  
quad_model = LinearRegression()
quad_model.fit(X_squared, y)
y_pred_quad = quad_model.predict(X_squared)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred_linear, color='red', label='Linear Regression')
plt.plot(np.sort(X, axis=0), y_pred_quad[np.argsort(X, axis=0)][:,0], color='green', label='Quadratic Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Linear vs Quadratic Regression')
plt.legend()
plt.show()

print(f"Koefisien linear: {linear_model.coef_[0]}")
print(f"Intercept linear: {linear_model.intercept_}")

mse_linear = mean_squared_error(y, y_pred_linear)
rms_linear = np.sqrt(mse_linear)
print(f"MSE Linear Regression: {mse_linear}")
print(f"RMS Linear Regression: {rms_linear}")

print()
print("Testing Linear Regression")
test_hours = [[2], [4], [6]]  
predicted_scores_linear = linear_model.predict(test_hours)
print("Predicted Scores (Linear Regression):")
for i, hours in enumerate(test_hours):
    print(f"Hours Studied: {hours[0]}, Predicted Score: {predicted_scores_linear[i]}")

print()
print(f"Koefisien quadratic: {quad_model.coef_[0]}")
print(f"Intercept quadratic: {quad_model.intercept_}")

mse_quad = mean_squared_error(y, y_pred_quad)
rms_quad = np.sqrt(mse_quad)
print(f"MSE Quadratic Regression: {mse_quad}")
print(f"RMS Quadratic Regression: {rms_quad}")

print()
print("Testing Quadratic Regression")
test_hours_squared = np.array(test_hours) ** 2
predicted_scores_quad = quad_model.predict(test_hours_squared)
print("Predicted Scores (Quadratic Regression):")
for i, hours in enumerate(test_hours):
    print(f"Hours Studied: {hours[0]}, Predicted Score: {predicted_scores_quad[i]}")
