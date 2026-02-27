# forecast.py
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_features(
    factor_rets: pd.DataFrame,
    port_ret: pd.Series,
    lookback: int = 20,
    lags=(1, 2, 3, 5),
    windows=(5, 20),
):
    """
    Build supervised ML dataset:
      X = lagged/rolling features from factor returns + portfolio vol
      y = next-day portfolio return (port_ret shifted by -1)

    Returns:
      X (DataFrame), y (Series)
    """
    df = pd.concat([factor_rets.copy(), port_ret.rename("port")], axis=1).dropna()

    # Lag factor returns
    for col in factor_rets.columns:
        for L in lags:
            df[f"{col}_lag{L}"] = df[col].shift(L)

    # Rolling mean/vol of factors (limited by lookback)
    for col in factor_rets.columns:
        for w in windows:
            w2 = min(w, lookback)
            df[f"{col}_rollmean{w2}"] = df[col].rolling(w2).mean()
            df[f"{col}_rollvol{w2}"] = df[col].rolling(w2).std()

    # Rolling vol of portfolio itself
    for w in windows:
        w2 = min(w, lookback)
        df[f"port_rollvol{w2}"] = df["port"].rolling(w2).std()

    # Target: next day return
    df["y_next"] = df["port"].shift(-1)

    df = df.dropna()
    y = df["y_next"]
    X = df.drop(columns=["port", "y_next"])
    return X, y


def time_split(X: pd.DataFrame, y: pd.Series, train_frac=0.70, val_frac=0.15):
    """
    Simple time-based split: train / val / test.
    """
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_val = X.iloc[n_train : n_train + n_val]
    y_val = y.iloc[n_train : n_train + n_val]

    X_test = X.iloc[n_train + n_val :]
    y_test = y.iloc[n_train + n_val :]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_mean_models(X_train, y_train, random_state=0):
    """
    Mean (point) prediction models.
    Returns dict: name -> fitted model
    """
    models = {}

    # Baseline: Ridge
    models["Ridge"] = Ridge(alpha=1.0).fit(X_train, y_train)

    # ML: Gradient Boosting (nonlinear)
    models["GBR"] = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    ).fit(X_train, y_train)

    return models


def train_quantile_models(X_train, y_train, qs=(0.05, 0.50, 0.95), alpha=1e-4):
    """
    Quantile regression models for VaR-ish outputs.
    Returns dict: "Q5"/"Q50"/"Q95" -> fitted model
    """
    out = {}
    for q in qs:
        name = f"Q{int(q*100)}"
        out[name] = QuantileRegressor(quantile=q, alpha=alpha, solver="highs").fit(X_train, y_train)
    return out


def evaluate_mean_model(model, X, y):
    pred = model.predict(X)

    mae = float(np.mean(np.abs(y.values - pred)))

    mse = float(np.mean((y.values - pred) ** 2))
    rmse = float(np.sqrt(mse))

    dir_acc = float(np.mean((pred > 0) == (y.values > 0)))

    return {"MAE": mae, "RMSE": rmse, "DirectionAcc": dir_acc}


def choose_best_by_val(models: dict, X_val, y_val):
    """
    Pick model with lowest val RMSE.
    """
    best_name = None
    best_rmse = float("inf")
    best_model = None

    for name, m in models.items():
        rmse = evaluate_mean_model(m, X_val, y_val)["RMSE"]
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = m

    return best_name, best_model, best_rmse


def approx_dist_from_quantiles(q05, q50, q95, n=8000, seed=0):
    """
    Turn (Q5, Q50, Q95) into an approximate normal distribution
    by mapping Q95-Q05 to ~3.29 sigma (normal).
    """
    rng = np.random.default_rng(seed)
    sigma = max(1e-8, (q95 - q05) / 3.29)
    return rng.normal(loc=q50, scale=sigma, size=n)