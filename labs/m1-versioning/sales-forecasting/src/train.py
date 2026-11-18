
import argparse
import json
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROC = Path("data/processed")

MODELS = {
    "random_forest": RandomForestRegressor,
    "gradient_boost": GradientBoostingRegressor,
    "decision_tree": DecisionTreeRegressor,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    # Charger les données préparées
    X_train = pd.read_parquet(PROC / "X_train.parquet")
    X_test = pd.read_parquet(PROC / "X_test.parquet")
    y_train = pd.read_parquet(PROC / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(PROC / "y_test.parquet").squeeze()

    # Construire le modèle
    if args.model == "random_forest":
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1
        )

    elif args.model == "gradient_boost":
        model = GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=42
        )

    else:
        model = DecisionTreeRegressor(
            max_depth=args.max_depth,
            random_state=42
        )

    # Configuration MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("sales-forecasting")

    with mlflow.start_run(run_name=args.model):

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Effectuer les prédictions
        predictions = model.predict(X_test)

        # Calculer les métriques
        metrics = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
            "R2": r2_score(y_test, predictions)
        }

        # Enregistrer les paramètres et métriques dans MLflow
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_metrics(metrics)

        # Sauvegarder les métriques pour DVC
        REPORTS = Path("reports")
        REPORTS.mkdir(parents=True, exist_ok=True)

        with open(REPORTS / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="sales_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"Modèle : {args.model}")
        print(f"Métriques : {metrics}")


if __name__ == "__main__":
    main()
