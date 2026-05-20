import optuna
import subprocess

def objective(trial):
    # Suggest parameters
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
        "window_size": trial.suggest_int("window_size", 5, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)
    }

    # Run train.py as a subprocess
    cmd = [
        "python", "train.py",
        str(params["max_depth"]),
        str(params["min_samples_split"]),
        str(params["window_size"]),
        str(params["min_samples_leaf"])
    ]
    
    # Capture the printed MSE from train.py
    result = subprocess.run(cmd, capture_output=True, text=True)
    mse = float(result.stdout.strip())
    print(mse)
    return mse

study = optuna.create_study(
    storage="sqlite:///db.sqlite3",  # This saves the data to a file
    study_name="study-1",
    direction="minimize",
    load_if_exists=True
)
study.optimize(objective, n_trials=100)
print(f"Best: {study.best_params}")
