import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def train_and_evaluate(max_depth, min_samples_split, window_size, min_samples_leaf):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    time_series = pd.read_csv(url)['Temp'].values

    # Dynamic Sliding Window
    X_seq, y_seq = [], []
    for i in range(len(time_series) - window_size):
        X_seq.append(time_series[i : i + window_size])
        y_seq.append(time_series[i + window_size])
    X_arr, y_arr = np.array(X_seq), np.array(y_seq)

    # Cross-validation
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []

    for train_index, test_index in tscv.split(X_arr):
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_arr[train_index], y_arr[train_index])
        mse_scores.append(mean_squared_error(y_arr[test_index], model.predict(X_arr[test_index])))

    return np.mean(mse_scores)

if __name__ == "__main__":
    # Same arg structure as your Katib script
    mse = train_and_evaluate(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    print(mse) # Output only the metric
