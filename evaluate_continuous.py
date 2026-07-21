"""Pilot evaluation of continuous treatment models."""

import numpy as np
from src.datasets.synthetic_loader import SyntheticDoseResponseDataset
from src.models.neural.dragonnet_continuous import DragonNetContinuous
from src.models.neural.tarnet_continuous import TARNetContinuous
from src.metrics.continuous_metrics import (
    mse_continuous,
    rmse_continuous,
    pearson_correlation,
    dose_response_curve,
)


def eval_continuous_model(model, X_test, T_test, Y_test, tau_true, model_name):
    """Evaluate a continuous treatment model."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    # Predict CATE
    cate_pred = model.predict_cate(X_test, T_test)

    # Metrics
    mse = mse_continuous(tau_true, cate_pred)
    rmse = rmse_continuous(tau_true, cate_pred)
    corr = pearson_correlation(tau_true, cate_pred)

    print(f"CATE MSE:  {mse:.6f}")
    print(f"CATE RMSE: {rmse:.6f}")
    print(f"CATE Corr: {corr:.6f}")
    print(f"CATE range: [{cate_pred.min():.4f}, {cate_pred.max():.4f}]")
    print(f"True CATE range: [{tau_true.min():.4f}, {tau_true.max():.4f}]")

    return {"mse": mse, "rmse": rmse, "corr": corr}


def main():
    print("Loading synthetic continuous treatment dataset...")
    ds_train = SyntheticDoseResponseDataset(scenario="linear", n_samples=500, split="train")
    ds_test = SyntheticDoseResponseDataset(scenario="linear", n_samples=500, split="test")

    # Extract data
    X_train = np.array([ds_train[i]["features"].numpy() for i in range(len(ds_train))])
    T_train = np.array([ds_train[i]["treatment"] for i in range(len(ds_train))])
    Y_train = np.array([ds_train[i]["outcome"] for i in range(len(ds_train))])

    X_test = np.array([ds_test[i]["features"].numpy() for i in range(len(ds_test))])
    T_test = np.array([ds_test[i]["treatment"] for i in range(len(ds_test))])
    Y_test = np.array([ds_test[i]["outcome"] for i in range(len(ds_test))])
    tau_true = np.array([ds_test[i]["cate_true"] for i in range(len(ds_test))])

    print(f"\nTrain: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"Test: {len(X_test)} samples")
    print(f"Treatment range: [{T_train.min():.3f}, {T_train.max():.3f}]")

    config = {
        "seed": 42,
        "hidden_dim": 150,
        "treatment_dim": 32,
        "n_layers": 3,
        "dropout": 0.1,
        "lr": 0.01,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "max_epochs": 50,
        "patience": 10,
        "val_fraction": 0.1,
        "alpha": 0.5,
    }

    results = {}

    # DragonNetContinuous
    print("\n" + "="*60)
    print("Training DragonNetContinuous...")
    print("="*60)
    model1 = DragonNetContinuous(config)
    model1.fit(X_train, T_train, Y_train)
    results["DragonNetContinuous"] = eval_continuous_model(model1, X_test, T_test, Y_test, tau_true, "DragonNetContinuous")

    # TARNetContinuous
    config["alpha"] = 0.0  # TARNetContinuous doesn't use alpha
    print("\n" + "="*60)
    print("Training TARNetContinuous...")
    print("="*60)
    model2 = TARNetContinuous(config)
    model2.fit(X_train, T_train, Y_train)
    results["TARNetContinuous"] = eval_continuous_model(model2, X_test, T_test, Y_test, tau_true, "TARNetContinuous")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")


if __name__ == "__main__":
    main()
