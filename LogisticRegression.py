import numpy as np


x = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [2.0, 2.0], [2.5, 1.5]])
y = np.array([0, 0, 1, 1, 1])


def find_slope(w,y, b,x):
    n = len(y)
    prediction = predict(w, x, b)
    dw = (1/n) * x.T.dot(prediction - y)
    db = np.mean(prediction - y)
    return dw, db

def cost_function(x, y, w, b):
    prediction = predict(w, x, b)
    return - (np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction)))
    

def classify(w, x, b, threshold=0.5):
    probabilities = predict(w, x, b)
    return (probabilities >= threshold).astype(int)

    
def predict(w, x , b):
    z = b + np.dot(x, w)
    return 1 / (1 + np.exp(-z))

lr = 0.01 #learning rate 
epochs = 10000 #iteration
w, b = np.zeros(x.shape[1]), 0.0 


for i in range(epochs):
    dw, db = find_slope(w, y, b, x)
    w = w - lr * dw
    b = b - lr * db
    
    
    
print("w and b : ", w, b)
print("cost error: ", cost_function(x, y, w, b))
print("Predicition: ", classify(w, x, b))
print("Actual values: ", y)

