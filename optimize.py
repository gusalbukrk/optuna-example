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
    
    # Capture the comma-separated metrics string
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Split the output string back into individual values
    metrics = result.stdout.strip().split(",")
    
    train_mse = float(metrics[0])
    test_mse = float(metrics[1])
    train_mae = float(metrics[2])
    test_mae = float(metrics[3])

    # Save extra metrics into the Optuna DB file so they show up on your dashboard
    trial.set_user_attr("train_mse", train_mse)
    trial.set_user_attr("train_mae", train_mae)
    trial.set_user_attr("test_mae", test_mae)

    print(f"Trial {trial.number} -> Test MSE: {test_mse} | Test MAE: {test_mae}")
    
    # Return ONLY test_mse so Optuna ignores the others when making predictions
    return test_mse

sampler = optuna.samplers.TPESampler(
    multivariate=True, 
    seed=42,
)

study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="study-3",
    direction="minimize",
    sampler=sampler,
    load_if_exists=True
)
study.optimize(objective, n_trials=20)
print(f"Best: {study.best_params}")
