import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])


# model for computing
def compute_model(x, w, b):
    return w * x + b


def model_error(x, y, w, b, alpha=0, rg=None):
    y_predictions = compute_model(x, w, b)
    mse = np.mean((y - y_predictions) ** 2)

    if rg == "l1":
        regularization_penalty = alpha * np.sum(np.abs(w))
    elif rg == "l2":
        regularization_penalty = alpha * np.sum(w**2)
    else:
        regularization_penalty = 0

    return mse + regularization_penalty


def gradient_descent(x, y, w, b, alpha, rg):
    n = len(x)
    prediction = compute_model(x, w, b)

    if rg == "l1":
        dw = -(1 / n) * x.T.dot(y - prediction) + alpha * np.sign(w)
    elif rg == "l2":
        dw = -(1 / n) * x.T.dot(y - prediction) + 2 * alpha * w
    else:
        dw = -(1 / n) * x.T.dot(y - prediction)

    db = -np.mean(y - prediction)
    return dw, db


def update_parameters(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    return w, b


# updates the w and b
def update_parameters(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    return w, b


lr = 0.01  # learning rate
epochs = 100000  # iteration
w, b = 0.0, 0.0
alpha= 0.1
rg=""

for i in range(epochs):
    dw, db = gradient_descent(x, y, w, b, alpha, rg)
    w, b = update_parameters(w, b, dw, db, lr)
    if i % 100 == 0:
        cost = model_error(x, y, w, b, alpha, rg)

y_predictions = compute_model(x, w, b)
error = model_error(x, y, w, b)

print("This is the y_predictions: ", y_predictions)
print("This is the error: ", error)
print("\nFinal parameters: w =", w, ", b =", b)
