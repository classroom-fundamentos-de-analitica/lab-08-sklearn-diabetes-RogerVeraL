import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def read_data():
    train_data = pd.read_csv('train_dataset.csv')
    test_data = pd.read_csv('test_dataset.csv')
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"Entrenamiento - MSE: {mse_train}, MAE: {mae_train}, R²: {r2_train}")
    print(f"Testeo - MSE: {mse_test}, MAE: {mae_test}, R²: {r2_test}")

def save_model(model):
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

def main():
    X_train, X_test, y_train, y_test = read_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, y_train, X_test, y_test)
    save_model(model)

if __name__ == '__main__':
    main()
