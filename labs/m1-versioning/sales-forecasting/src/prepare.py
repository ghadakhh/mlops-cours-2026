import pandas as pd
from pathlib import Path

RAW = Path("data/raw/Sales_MLOps_Synthetic_2020_2022.xlsx")
PROC = Path("data/processed")

def main():
    PROC.mkdir(parents=True, exist_ok=True)

    # Charger les données
    df = pd.read_excel(RAW)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Date"] = df["Order Date"].dt.normalize()

    # Une seule ligne par jour
    daily = (
        df.groupby("Date")["Daily_Sales"]
        .first()
        .reset_index()
        .sort_values("Date")
    )

    # Variables temporelles
    daily["Year"] = daily["Date"].dt.year
    daily["Month"] = daily["Date"].dt.month
    daily["Day"] = daily["Date"].dt.day
    daily["Weekday"] = daily["Date"].dt.day_of_week
    daily["DayOfYear"] = daily["Date"].dt.day_of_year
    daily["Is_Weekend"] = (daily["Weekday"] >= 5).astype(int)
    daily["Trend"] = range(len(daily))

    features = [
        "Year", "Month", "Day", "Weekday",
        "DayOfYear", "Is_Weekend", "Trend"
    ]

    X = daily[features]
    y = daily["Daily_Sales"]

    # Séparation chronologique
    train_mask = daily["Date"] < "2022-01-01"

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    # Sauvegarder les données
    X_train.to_parquet(PROC / "X_train.parquet", index=False)
    X_test.to_parquet(PROC / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(PROC / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(PROC / "y_test.parquet", index=False)

    print("Préparation terminée !")
    print(f"Nombre total de jours : {len(daily)}")
    print(f"Train : {len(X_train)} jours (2020-2021)")
    print(f"Test : {len(X_test)} jours (2022)")
    print(f"Variables : {features}")

if __name__ == "__main__":
    main()
