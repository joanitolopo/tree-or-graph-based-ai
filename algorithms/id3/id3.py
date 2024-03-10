import numpy as np

class IDDecisionTree:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass



if __name__=="__main__":
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])

    model = IDDecisionTree()
    model.fit(X_train, y_train)

    X_test = np.array([[0, 1], [1, 0]])
    prediction = model.predict(X_train)