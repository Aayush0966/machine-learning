import numpy as np

def gini_impurity(y):
    classes, count = np.unique(y, return_counts=True)
    probs = count / len(y)
    return 1 - np.sum(probs ** 2)


def info_gain(y, feature):
    parent_gain = gini_impurity(y)
    best_gain = -1
    best_thresh = None

    unique_features = np.unique(feature)
    if len(unique_features) == 1:
        return 0, unique_features[0]

    thresholds = (unique_features[:-1] + unique_features[1:]) / 2

    for t in thresholds:
        left_y = y[feature <= t]
        right_y = y[feature > t]
        if len(left_y) == 0 or len(right_y) == 0:
            continue

        weighted_gini = (len(left_y) / len(y)) * gini_impurity(left_y) + \
                        (len(right_y) / len(y)) * gini_impurity(right_y)

        gain = parent_gain - weighted_gini

        if gain > best_gain:
            best_thresh = t
            best_gain = gain

    return best_gain, best_thresh


def best_split(y, X):
    best_gain = -1
    best_feature_idx = None
    best_threshold = None

    for i in range(X.shape[1]):
        gain, thresh = info_gain(y, X[:, i])
        if gain > best_gain:
            best_gain = gain
            best_feature_idx = i
            best_threshold = thresh

    return best_feature_idx, best_threshold


def build_tree(y, X):
    if len(np.unique(y)) == 1:
        return int(y[0])

    idx, threshold = best_split(y, X)
    if idx is None:
        return int(np.round(np.mean(y)))

    left_mask = X[:, idx] <= threshold
    right_mask = X[:, idx] > threshold

    tree = {
        "feature_idx": idx,
        "threshold": threshold,
        "branches": {
            "left": build_tree(y[left_mask], X[left_mask]),
            "right": build_tree(y[right_mask], X[right_mask]),
        }
    }
    return tree


def predict(tree, sample):
    if isinstance(tree, int):
        return tree
    idx = tree["feature_idx"]
    threshold = tree["threshold"]
    if sample[idx] <= threshold:
        return predict(tree["branches"]["left"], sample)
    else:
        return predict(tree["branches"]["right"], sample)


def predict_all(tree, X):
    return np.array([predict(tree, x) for x in X])

X = np.array([
    [150, 45],
    [155, 47],
    [160, 52],
    [162, 55],
    [165, 58],
    [168, 63],
    [170, 66],
    [172, 70],
    [175, 75],
    [178, 77],
    [180, 80],
    [182, 83],
    [185, 88],
    [188, 92],
    [190, 95],
    [192, 98],
    [195, 105],
    [198, 110],
    [200, 115],
    [205, 118],
    [160, 70],   # noisy point - short but heavy
    [190, 60],   # tall but light
    [168, 50],   # medium but light
    [175, 95],   # medium but heavy
])

# Labels with noise — not strictly linear relation
y = np.array([
    0, 0, 0, 0, 0, 
    1, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1,
    0, 0, 0, 1   # noisy mismatches
])

# Split into train/test
X_train, X_test = X[:18], X[18:]
y_train, y_test = y[:18], y[18:]

tree = build_tree(y_train, X_train)
y_pred = predict_all(tree, X_test)

print("Predictions:", y_pred)
print("Actual:", y_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)



class RandomForest:
    def __init__(self, n_trees=5, max_features=None):
        self.n_trees = n_trees
        self.trees = []
        self.max_features = max_features
        
    
    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_trees):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]
            
            if self.max_features is not None:
                feature_idx = np.random.choice(X.shape[1], self.max_features, replace=False)
                X_sample = X_sample[:, feature_idx]
            else:
                feature_idx = np.arange(X.shape[1])
                
            
            tree = build_tree(y_sample, X_sample)
            self.trees.append((tree, feature_idx))
    
    def predict(self, X):
        tree_preds = []
        
        for tree, feature_idx in self.trees:
            X_subsets = X[:, feature_idx]
            preds = predict_all(tree, X_subsets)
            tree_preds.append(preds)
        
        tree_preds = np.array(tree_preds)
        y_pred = []
        for i in range(X.shape[0]):
            counts = np.bincount(tree_preds[:, i])
            y_pred.append(np.argmax(counts))
        
        return np.array(y_pred)
        
        
        
        
X_train, X_test = X[:18], X[18:]
y_train, y_test = y[:18], y[18:]

rf = RandomForest(n_trees=100, max_features=2)  
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Predictions:", y_pred)
print("Actual:", y_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
