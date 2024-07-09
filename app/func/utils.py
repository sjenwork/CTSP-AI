from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
import json
import pandas as pd
import numpy as np


def load_my_model():
    targets = ["PM2.5", "O3"]
    models = {}

    for target in targets:
        model = load_model(f"{target}訓練/{target.replace('.','')}_prediction_model.h5")
        optimizer = RMSprop(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss="mean_squared_error")
        models[target] = model
    return models


def create_sequences(data, target):
    X = data.to_numpy()
    X = X.reshape(1, *X.shape)

    return X


def pred(data_all):
    data = pd.DataFrame(data_all["TimeSeries"]).set_index(["TimePoint", "TimeIndex"])
    mean = pd.Series(data_all["Mean"])
    std = pd.Series(data_all["Std"])
    data_anomaly = (data - mean) / std

    history_data = data_anomaly.iloc[:168]
    prediction_data = data_anomaly.iloc[168:]
    features = list(data_anomaly.columns)
    targets = ["PM2.5", "O3"]
    results = []
    for target in targets:
        X = create_sequences(history_data, target=target)
        # 預測
        y_pred = models[target].predict(X)

        result_df = data_anomaly.loc[:, [target]].copy().rename(columns={target: f"{target}_true"})
        result_df[f"{target}_pred"] = None
        result_df[f"{target}_pred"].iloc[-8:] = y_pred[0]
        result_df = result_df * std[target] + mean[target]
        results.append(result_df)

    result_merged = pd.concat(results, axis=1)
    result_merged = result_merged.astype(object).where(pd.notnull(result_merged), None)
    result_merged = result_merged.reset_index()
    res = result_merged.to_dict(orient="records")
    return res


models = load_my_model()

if __name__ == "__main__":
    with open("ctsp/example_input.json", "r") as f:
        data_all = json.loads(f.read())

    res = pred(data_all)
