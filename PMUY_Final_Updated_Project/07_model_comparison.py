"""
Pooled PMUY regression with chronological validation.

This script compares:
1. Pooled Linear Regression with district indicators
2. Pooled Polynomial Regression (degree 2 in time)
3. Pooled Ridge Regression with district indicators

It uses expanding-window validation: each held-out financial year is
predicted using only earlier years. It also reports training R2 separately.
A positive training R2 must not be presented as a positive test R2.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
DATA_XLSX = ROOT / "data" / "West_Bengal_PMUY_NFHS5_Combined.xlsx"
DATA_CSV = ROOT / "data" / "West_Bengal_PMUY_NFHS5_Combined.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

YEAR_COLUMNS = [
    "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22"
]
YEAR_TO_INDEX = {year: i for i, year in enumerate(YEAR_COLUMNS)}
MODEL_NAMES = ["Pooled Linear", "Pooled Polynomial", "Pooled Ridge"]


def load_data():
    if DATA_XLSX.exists():
        wide = pd.read_excel(DATA_XLSX)
    elif DATA_CSV.exists():
        wide = pd.read_csv(DATA_CSV)
    else:
        raise FileNotFoundError(
            "Place West_Bengal_PMUY_NFHS5_Combined.xlsx or .csv inside data/."
        )

    required = ["District"] + YEAR_COLUMNS
    missing = [c for c in required if c not in wide.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Do not silently replace missing values with zero.
    wide[YEAR_COLUMNS] = wide[YEAR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if wide[YEAR_COLUMNS].isna().any().any():
        raise ValueError("Missing/non-numeric PMUY yearly values found. Check the source data.")

    long = wide.melt(
        id_vars="District",
        value_vars=YEAR_COLUMNS,
        var_name="Year",
        value_name="Connections"
    )
    long["YearIndex"] = long["Year"].map(YEAR_TO_INDEX).astype(int)
    return wide, long


def make_model(model_name):
    if model_name == "Pooled Linear":
        features = ColumnTransformer(
            transformers=[
                ("district", OneHotEncoder(handle_unknown="ignore"), ["District"]),
                ("year", "passthrough", ["YearIndex"])
            ]
        )
        estimator = LinearRegression()

    elif model_name == "Pooled Polynomial":
        features = ColumnTransformer(
            transformers=[
                ("district", OneHotEncoder(handle_unknown="ignore"), ["District"]),
                ("year_poly", PolynomialFeatures(degree=2, include_bias=False), ["YearIndex"])
            ]
        )
        estimator = LinearRegression()

    else:
        features = ColumnTransformer(
            transformers=[
                ("district", OneHotEncoder(handle_unknown="ignore"), ["District"]),
                ("year", StandardScaler(), ["YearIndex"])
            ]
        )
        estimator = Ridge(alpha=1.0)

    return Pipeline([("features", features), ("model", estimator)])


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }


def main():
    wide, data = load_data()
    print(f"Dataset: {len(wide)} districts x {len(YEAR_COLUMNS)} complete years")
    print("Note: 2022-23_Apr-Feb is partial and is excluded.")
    print("Recorded zero values for 2020-21 are retained and flagged for source verification.")

    all_predictions = []
    yearly_metrics = []
    training_metrics = []

    # Expanding-window validation: predict 2019-20, 2020-21, and 2021-22.
    # Every test fold is a complete year across all districts.
    for model_name in MODEL_NAMES:
        for test_year_index in [3, 4, 5]:
            train = data[data["YearIndex"] < test_year_index].copy()
            test = data[data["YearIndex"] == test_year_index].copy()

            model = make_model(model_name)
            X_train = train[["District", "YearIndex"]]
            y_train = train["Connections"]
            X_test = test[["District", "YearIndex"]]
            y_test = test["Connections"]

            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_result = metrics(y_train, train_pred)
            test_result = metrics(y_test, test_pred)

            training_metrics.append({
                "Model": model_name,
                "Training_Through": YEAR_COLUMNS[test_year_index - 1],
                "Training_R2": train_result["R2"],
                "Training_MAE": train_result["MAE"],
                "Training_RMSE": train_result["RMSE"]
            })

            yearly_metrics.append({
                "Model": model_name,
                "Test_Year": YEAR_COLUMNS[test_year_index],
                "Training_Years": f"{YEAR_COLUMNS[0]} to {YEAR_COLUMNS[test_year_index - 1]}",
                "Test_R2": test_result["R2"],
                "Test_MAE": test_result["MAE"],
                "Test_RMSE": test_result["RMSE"]
            })

            fold = test[["District", "Year", "YearIndex", "Connections"]].copy()
            fold["Model"] = model_name
            fold["Predicted_Connections"] = test_pred
            all_predictions.append(fold)

    predictions = pd.concat(all_predictions, ignore_index=True)
    yearly = pd.DataFrame(yearly_metrics)
    training = pd.DataFrame(training_metrics)

    # Overall out-of-time R2 calculated across all 69 held-out district-year rows.
    overall_rows = []
    for model_name in MODEL_NAMES:
        part = predictions[predictions["Model"] == model_name]
        m = metrics(part["Connections"], part["Predicted_Connections"])
        overall_rows.append({
            "Model": model_name,
            "Out_of_Time_R2": m["R2"],
            "Out_of_Time_MAE": m["MAE"],
            "Out_of_Time_RMSE": m["RMSE"],
            "Held_Out_Observations": len(part)
        })
    overall = pd.DataFrame(overall_rows)

    # Separate final-year holdout: train on first five years, test on 2021-22.
    final_rows = []
    for model_name in MODEL_NAMES:
        train = data[data["YearIndex"] < 5]
        test = data[data["YearIndex"] == 5]
        model = make_model(model_name)
        model.fit(train[["District", "YearIndex"]], train["Connections"])
        pred = model.predict(test[["District", "YearIndex"]])
        m = metrics(test["Connections"], pred)
        final_rows.append({
            "Model": model_name,
            "Test_Year": "2021-22",
            "Training_Years": "2016-17 to 2020-21",
            "Test_R2": m["R2"],
            "Test_MAE": m["MAE"],
            "Test_RMSE": m["RMSE"]
        })
    final_year = pd.DataFrame(final_rows)

    training.to_csv(OUT / "pooled_training_metrics.csv", index=False)
    yearly.to_csv(OUT / "pooled_chronological_yearly_metrics.csv", index=False)
    predictions.to_csv(OUT / "pooled_chronological_predictions.csv", index=False)
    overall.to_csv(OUT / "pooled_chronological_overall_metrics.csv", index=False)
    final_year.to_csv(OUT / "pooled_final_year_holdout_metrics.csv", index=False)

    print("\nTRAINING R2 (in-sample; not test performance)")
    print(training.groupby("Model")["Training_R2"].mean().round(4))
    print("\nCHRONOLOGICAL TEST RESULTS BY YEAR")
    print(yearly.round(4).to_string(index=False))
    print("\nOVERALL OUT-OF-TIME RESULTS (69 held-out observations)")
    print(overall.round(4).to_string(index=False))
    print("\nFINAL-YEAR HOLDOUT: train 2016-17 to 2020-21, test 2021-22")
    print(final_year.round(4).to_string(index=False))
    print(f"\nResults saved in: {OUT}")


if __name__ == "__main__":
    main()
