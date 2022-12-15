import numpy as np

class LogisticRegressionClassifier():
    """
    Class for logisttic regression
    """

    def __init__(self, lr=0.001, epochs=1000):
        """
        Logistic Regression Constructor

        Args:
            lr (float, optional): _description_. Defaults to 0.001.
            epochs (int, optional): _description_. Defaults to 1000.
        """
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        
    def loss_function(self,x):
        """
        Sigmoid loss function

        Args:
            x (_type_): Z value(mx+b)

        Returns:
            _type_: Probability
        """
        return 1/(1+np.exp(-x))
    
    def fit(self, X, y):
        """
        Train the model

        Args:
            X (_type_): Features
            y (_type_): Response variable
        """
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0

        for e in range(self.epochs):
            linear_pred = np.dot(X, self.w) + self.b
            pred = self.loss_function(linear_pred)

            dw = (1/n) * np.dot(X.T, (pred - y))
            db = (1/n) * np.sum(pred-y)

            self.w = self.w - self.lr*dw
            self.b = self.b - self.lr*db

    def predict(self, X):
        """
        Predict the Y

        Args:
            X (_type_): _description_

        Returns:
            _type_: Y-hat probability
        """
        linear_pred = np.dot(X, self.w) + self.b
        y_pred = self.loss_function(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred