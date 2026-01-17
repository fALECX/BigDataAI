import numpy as np

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = None
        self.split_left = None # categories only: split criteria left
        self.split_right = None # categories only: split criteria right
        self.left = None
        self.right = None

def subsets(unique_attributes):
    if unique_attributes == []:
        return [[]]
    x = subsets(unique_attributes[1:])
    return x + [[unique_attributes[0]] + y for y in x]

def binary_split(unique_attributes):
    attributes_left = [x for x in subsets(unique_attributes) if len(x) > 0 and len(x) < len(unique_attributes) / 2]
    if len(unique_attributes) % 2 == 0:
        attr_left = ([x for x in subsets(unique_attributes) if len(x) == len(unique_attributes) / 2])
        attributes_left += attr_left[:int(len(attr_left) / 2)]
    attributes_right = [list(set(unique_attributes) - set(x)) for x in attributes_left]
    return attributes_left, attributes_right

def gini_impurity(num, n_classes):
    return 1.0 - np.sum((num[c] / np.sum(num))**2 for c in range(n_classes))

def best_split(X, y, n_classes, max_features, average_thresholds=False):
    if max_features is None:
        features = list(range(X.shape[1])) # all attributes
    else:
        features = np.random.choice(X.shape[1], size=int(np.round(np.sqrt(X.shape[1]))), replace=False) # sqrt(n) attributes without replacement
    m = y.size # number of data points
    if m <= 1:
        return None, None, None, None
    num_parent = [np.sum(y == c) for c in range(n_classes)]
    best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
    best_idx, best_thr, best_split_left, best_split_right = None, None, None, None
    for idx in features:
        if isinstance(X[0, idx], str): # categorical attribute
            attr_left, attr_right = binary_split(list(set(X[:, idx])))
            for i in range(len(attr_left)):
                ind_left = np.isin(X[:, idx], attr_left[i])
                num_left = [0] * n_classes # start with none data points
                for c in range(n_classes):
                    num_left[c] = np.sum(y[ind_left] == c)
                num_right = np.array(num_parent.copy())
                num_right -= np.array(num_left)
                gini_left = gini_impurity(num_left, n_classes)
                gini_right = gini_impurity(num_right, n_classes)
                gini_index = (np.sum(num_left) * gini_left + np.sum(num_right) * gini_right) / m
                if gini_index < best_gini:
                    best_gini = gini_index
                    best_idx = idx
                    best_split_left = attr_left[i]
                    best_split_right = attr_right[i]
                    best_thr = None
        else: # numerical attribute
            thresholds, classes = zip(*sorted(zip(X[:, idx], y))) # sort thresholds/ classes pairs by threshold
            num_left = [0] * n_classes # start with none data points left
            num_right = num_parent.copy() # all data points right
            first_i = 1 if average_thresholds else 2
            for i in range(first_i, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                if thresholds[i] == thresholds[i - 1]:
                    continue
                gini_left = gini_impurity(num_left, n_classes)
                gini_right = gini_impurity(num_right, n_classes)
                gini_index = (np.sum(num_left) * gini_left + np.sum(num_right) * gini_right) / m
                if gini_index < best_gini:
                    best_gini = gini_index
                    best_idx = idx
                    if average_thresholds:
                        best_thr = (thresholds[i] + thresholds[i - 1]) / 2
                    else:
                        best_thr = thresholds[i]
                    best_split_left, best_split_right = None, None
    return best_idx, best_thr, best_split_left, best_split_right

def build_tree(X, y, n_classes, depth=0, max_depth=3, max_features=None):
    num_samples_per_class = [np.sum(y == i) for i in range(n_classes)]
    predicted_class = np.argmax(num_samples_per_class)    
    node = Node(predicted_class=predicted_class)
    if depth < max_depth:
        idx, thr, split_left, split_right = best_split(X, y, n_classes, max_features)
        if idx is not None:
            if thr is not None:
                indices_left = X[:, idx] < thr
                node.threshold = thr
            else:
                indices_left = np.isin(X[:, idx], split_left)                
                node.split_left = split_left
                node.split_right = split_right
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]        
            node.feature_index = idx
            node.left = build_tree(X_left, y_left, n_classes, depth + 1, max_depth)
            node.right = build_tree(X_right, y_right, n_classes, depth + 1, max_depth)
    return node

def predict(rootNode, X):
    preds = []
    for x in X:
        node = rootNode
        while node.left:
            if (node.split_left and np.isin(x[node.feature_index], node.split_left)) or (node.threshold and x[node.feature_index] < node.threshold):
                node = node.left
            else:
                node = node.right
        preds.append(node.predicted_class)
    return preds

def print_tree(node, depth, feature_names, class_names):
    if not node.left: # leaf node
        print(" " * depth + "|-- class: " + class_names[node.predicted_class], sep="")
    else:
        if node.split_left:
            print(" " * depth + "|-- " + feature_names[node.feature_index] + " in " + str(node.split_left), sep="")
        else:
            print(" " * depth + "|-- " + feature_names[node.feature_index] + " < " + str(node.threshold), sep="")
        print_tree(node.left, depth + 1, feature_names, class_names)
        if node.split_right:
            print(" " * depth + "|-- " + feature_names[node.feature_index] + " in " + str(node.split_right), sep="")
        else:
            print(" " * depth + "|-- " + feature_names[node.feature_index] + " >= " + str(node.threshold), sep="")
        print_tree(node.right, depth + 1, feature_names, class_names)
