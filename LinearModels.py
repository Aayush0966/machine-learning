import numpy as np


class LinearRegression():
    def best_fit(self, x, y, rg):
        X = np.array(x)
        Y = np.array(y)
        
        if X.ndim != 2:
            raise ValueError(f"X should be 2-dimensional (samples, features). Got shape: {X.shape}")
        if Y.ndim != 1:
            raise ValueError(f"Y should be 1-dimensional (samples,). Got shape: {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Number of samples in X and Y must match. Got X.shape[0]={X.shape[0]}, Y.shape[0]={Y.shape[0]}")

        tolerance = 1e-6
        previous_cost = float('inf')
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std[x_std == 0] = tolerance  # in case std is 0
        x_scaled = (X - x_mean) / x_std  # scaling because some features are large in size and some are small
        
        w_scaled = np.zeros((x_scaled.shape[1], ))  
        b_scaled = 0
        lr = 0.01
        alpha=0.01
        epochs = 100000
        
        for i in range(epochs):
            dw, db = self.find_slope(w_scaled, b_scaled, x_scaled, Y, alpha, rg)
            w_scaled = w_scaled - lr * dw
            b_scaled = b_scaled - lr * db
            current_cost = self.cost_function(w_scaled, b_scaled, x_scaled, Y, alpha, rg)

            if previous_cost != float('inf'):
                if abs(previous_cost - current_cost) / previous_cost < 1e-6:
                    print("Early stopping at iteration:", i)
                    break
            
            previous_cost = current_cost
        
        # Convert weights back to original scale
        w_original = w_scaled / x_std
        b_original = b_scaled - np.sum(w_original * x_mean)

        # Clean up near-zero values
        w_original = np.where(np.isclose(w_original, 0), 0, w_original)
        if np.isclose(b_original, 0):
            b_original = 0

        # Calculate final cost with original weights
        final_cost = self.cost_function(w_original, b_original, X, Y, alpha, "")
        if np.isclose(final_cost, 0):
            final_cost = 0

        
        return w_original, b_original, final_cost

    def find_slope(self, w, b, x, y, alpha, rg):
        n = len(x)
        prediction = self.predict(w, x ,b)

        if rg == "l1":
            dw = -(1 / n) * x.T.dot(y - prediction) + alpha * np.sign(w)
        elif rg == "l2":
            dw = -(1 / n) * x.T.dot(y - prediction) + 2 * alpha * w
        else:
            dw = -(1 / n) * x.T.dot(y - prediction)

        db = -np.mean(y - prediction)
        return dw, db

    def cost_function(self, w, b, x, y, alpha, rg):
        y_predictions = self.predict(w, x, b)
        mse = np.mean((y - y_predictions) ** 2)

        if rg == "l1":
            regularization_penalty = alpha * np.sum(np.abs(w))
        elif rg == "l2":
            regularization_penalty = alpha * np.sum(w**2)
        else:
            regularization_penalty = 0

        return mse + regularization_penalty
    
    def predict(self, w, x, b):
        return x.dot(w) + b



class LogisticRegression():
    def best_fit(self, x, y, rg):
        X = np.array(x)
        Y = np.array(y)
        
        if X.ndim != 2:
            raise ValueError(f"X should be 2-dimensional (samples, features). Got shape: {X.shape}")
        if Y.ndim != 1:
            raise ValueError(f"Y should be 1-dimensional (samples,). Got shape: {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Number of samples in X and Y must match. Got X.shape[0]={X.shape[0]}, Y.shape[0]={Y.shape[0]}")

        tolerance = 1e-6
        previous_cost = float('inf')
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std[x_std == 0] = tolerance
        x_scaled = (X - x_mean) / x_std
        
        w_scaled = np.zeros(x_scaled.shape[1])
        b_scaled = 0
        lr = 0.01
        epochs = 10000
        alpha=0.01
        
        for i in range(epochs):
            dw, db = self.find_slope(w_scaled, b_scaled, x_scaled, y, alpha, rg)
            w_scaled = w_scaled - lr * dw
            b_scaled = b_scaled - lr * db
            current_cost = self.cost_function(w_scaled,b_scaled,x_scaled,y, alpha, rg)
            
            if previous_cost != float("inf"):
                if abs(previous_cost - current_cost) / previous_cost < 1e-6:
                    print("Early stopping at iteration: ", i)
                    break
            
            previous_cost = current_cost
        
        w_orig = w_scaled / x_std
        b_orig = b_scaled - np.sum(w_orig * x_mean)
        
        w_orig = np.where(np.isclose(w_orig, 0), 0, w_orig)
        if np.isclose(b_orig, 0):
            b_orig = 0
        
        final_cost = self.cost_function(w_orig, b_orig, X, Y, alpha, "")
        if np.isclose(final_cost, 0):
            final_cost = 0
        
        return w_orig, b_orig, final_cost
                    
    
    def cost_function(self, w, b, x, y, alpha=0, rg=None):  
        prediction = self.predict(w, x, b)
        prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
        cross_entropy = - (np.mean(y * np.log(prediction) + (1-y) * np.log(1 - prediction)))
        
        if rg == "l1":
            regularization_penalty = alpha * np.sum(np.abs(w))
        elif rg == "l2":
            regularization_penalty = alpha * np.sum(w**2)
        else:
            regularization_penalty = 0
        
        return cross_entropy + regularization_penalty
        
    def find_slope(self, w, b, x, y, alpha=0, rg=None):  
        prediction = self.predict(w, x, b)
        dw_base = (1/len(y)) * x.T.dot(prediction - y)
        
        if rg == "l1":
            dw = dw_base + alpha * np.sign(w)
        elif rg == "l2":
            dw = dw_base + 2 * alpha * w
        else:
            dw = dw_base
        
        db = np.mean(prediction - y)
        return dw, db
    
    def predict(self, w, x, b):
        z = b + np.dot(x,w )
        return 1/ (1 + np.exp(-z))
    
    
    def classify(self, w, x, b, threshold=0.5):
        prediction = self.predict(w, x, b)
        return (prediction >= threshold).astype(int)