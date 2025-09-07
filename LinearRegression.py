import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

#model for computing 
def compute_model(x, w, b):
    return w * x + b

#MSE 
def model_error(x, y, w, b):
    n = len(x)
    y_predictions = compute_model(x, w, b)
    return 1 / (2 * n) * np.sum((y - y_predictions) ** 2)  

#provides the derivative w.r.t w and b
def gradient_descent(x, y, w, b):   
    n = len(x)
    prediction = compute_model(x, w, b)
    dw = -(1 / n) * x.T.dot(y - prediction)
    db = - np.mean(y - prediction)
    return dw, db

#updates the w and b
def update_parameters(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    return w, b

lr = 0.01 #learning rate 
epochs = 100000 #iteration
w, b = 0.0, 0.0 

for i in range(epochs):
    dw, db = gradient_descent(x, y, w, b)
    w, b = update_parameters(w, b, dw, db, lr)
    if i % 100 == 0:
        cost = model_error(x, y, w, b)

y_predictions = compute_model(x, w, b)
error = model_error(x, y, w, b)

print("This is the y_predictions: ", y_predictions)
print("This is the error: ", error)
print("\nFinal parameters: w =", w, ", b =", b)
