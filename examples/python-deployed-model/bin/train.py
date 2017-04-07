from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

if __name__ == '__main__':
    boston = load_boston()
    X = boston.data
    y = boston.target

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, '../models/model.pkl')
