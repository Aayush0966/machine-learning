from pyexpat.errors import XML_ERROR_MISPLACED_XML_PI
import numpy as np



class LinearRegression():
    def best_fit(self, x, y):
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
        
        w_scaled = np.zeros(x_scaled.shape[1])
        b_scaled = 0
        lr = 0.01
        epochs = 100000
        
        for i in range(epochs):
            dw, db = self.find_slope(w_scaled, b_scaled, x_scaled, Y)
            w_scaled = w_scaled - lr * dw
            b_scaled = b_scaled - lr * db
            current_cost = self.cost_function(w_scaled, b_scaled, x_scaled, Y)

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
        final_cost = self.cost_function(w_original, b_original, X, Y)
        if np.isclose(final_cost, 0):
            final_cost = 0

        
        return w_original, b_original, final_cost

    def find_slope(self, w, b, x, y):
        n = len(y)
        predictions =  x.dot(w) + b
        dw = - (1/n)  * x.T.dot(y - predictions)
        db = - np.mean(y - predictions)
        return dw, db
    

    def cost_function(self, w, b, x, y):
        n = len(y)
        predictions = x.dot(w) + b
        return (1/(2*n)) * np.sum((y - predictions) ** 2)
    
    def predict(self, w, x, b):
        return x.dot(w) + b



class LogisticRegression():
    def best_fit(self, x, y):
        X = np.array(x)
        Y = np.array(y)
        tolerance = 1e-6
        previous_cost = float('inf')
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std[x_std == 0] = tolerance
        x_scaled = (X - x_mean) / x_std
        
        w_scaled = np.zeros(x_scaled.shape[1])
        b_scaled = 0
        lr = 0.01
        epochs = 5000
        
        for i in range(epochs):
            dw, db = self.find_slope(w_scaled, b_scaled, x, y)
            
        
        
        
    
    def find_slope(self, w, b, x, y):
        prediction  = self.predict(w, b,x)
        dw =  (1/len(y)  )*  x.t.dot(prediction - y)
        db = np.mean(prediction - y)
        return dw, db
        
    
    def predict(self, w, b, x):
        z = b + np.dot(x,w )
        return 1/ (1 + np.exp(-z))