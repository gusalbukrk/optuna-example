import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def train_and_evaluate(max_depth, min_samples_split, window_size, min_samples_leaf):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    time_series = pd.read_csv(url)['Temp'].values

    # Dynamic Sliding Window
    X_seq, y_seq = [], []
    for i in range(len(time_series) - window_size):
        X_seq.append(time_series[i : i + window_size])
        y_seq.append(time_series[i + window_size])
    X_arr, y_arr = np.array(X_seq), np.array(y_seq)

    # Cross-validation setup
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Tracking metrics across folds
    train_mse_scores = []
    test_mse_scores = []
    train_mae_scores = []
    test_mae_scores = []

    for train_index, test_index in tscv.split(X_arr):
        X_train, X_test = X_arr[train_index], X_arr[test_index]
        y_train, y_test = y_arr[train_index], y_arr[test_index]

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Append scores per split
        train_mse_scores.append(mean_squared_error(y_train, train_preds))
        test_mse_scores.append(mean_squared_error(y_test, test_preds))
        train_mae_scores.append(mean_absolute_error(y_train, train_preds))
        test_mae_scores.append(mean_absolute_error(y_test, test_preds))

    # Compute final averages
    return (
        np.mean(train_mse_scores),
        np.mean(test_mse_scores),
        np.mean(train_mae_scores),
        np.mean(test_mae_scores)
    )

if __name__ == "__main__":
    train_mse, test_mse, train_mae, test_mae = train_and_evaluate(
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    )
    # Output metrics as a comma-separated string for optimize.py to parse easily
    print(f"{train_mse},{test_mse},{train_mae},{test_mae}")
