from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import RMSprop
from matplotlib import pyplot as plt
import json
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.max_colwidth", 30)
pd.set_option("display.max_columns", 400)
pd.options.display.width = 0


def fill_missing_values(df, columns):
    for column in columns:
        for date, group in df.groupby(df.index.date):
            # 當天有值的部分平均值（不包含0和負值）
            valid_values = group[column].replace([0, -np.inf, np.inf], np.nan)
            valid_values = valid_values[valid_values >= 0].dropna()
            daily_non_null_count = valid_values.count()
            if daily_non_null_count > 0:
                daily_mean = valid_values.mean()
            else:
                daily_mean = np.nan

            for i, row in group.iterrows():
                if pd.isna(row[column]) or row[column] < 0:
                    if daily_non_null_count > 0:
                        # 若當天有值，缺失值則補上當天的有值部分的平均值
                        df.at[i, column] = daily_mean
                    else:
                        # 若當天全部為缺失值，則補上前一天有值的日期之平均值
                        d = i.replace(hour=0, minute=0, second=0)
                        prev_day_start = d - pd.Timedelta(days=1)
                        while True:
                            prev_day_data = df.loc[
                                prev_day_start : prev_day_start + pd.Timedelta(hours=23, minutes=59, seconds=59)
                            ]
                            prev_day_values = prev_day_data[column].replace([0, -np.inf, np.inf], np.nan)
                            prev_day_values = prev_day_values[prev_day_values >= 0].dropna()
                            prev_day_non_null_count = prev_day_values.count()
                            if prev_day_non_null_count > 0:
                                prev_day_mean = prev_day_values.mean()
                                df.at[i, column] = prev_day_mean
                                break
                            prev_day_start -= pd.Timedelta(days=1)
    return df


def create_sequences(df, features, target, past_hours=168, future_hours=8):
    X, y = [], []
    for i in range(len(df) - past_hours - future_hours):
        X.append(df[features].iloc[i : i + past_hours].values)
        y.append(df[target].iloc[i + past_hours : i + past_hours + future_hours].values)
    return np.array(X), np.array(y)


# 讀取Excel檔案
file_path = "2020-2024_國安國小_17物種.xlsx"
df = pd.read_excel(file_path)

df = df.iloc[:500]
# 將 'TimePoint' 列轉換為日期時間格式並設定為索引
df["TimePoint"] = pd.to_datetime(df["TimePoint"], format="%Y/%m/%d %H:%M")
df.set_index("TimePoint", inplace=True)

columns = df.columns
features = ["PM2.5", "PM10", "P", "WD", "WS", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "SO2"]
# features = ['PM2.5', 'PM10', 'P', 'WD', 'WS', 'CH4',
#             'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3',]

# 填補缺失值
fill_missing_values(df, features)

df["WD_sin"] = np.sin(np.deg2rad(df["WD"]))
df["WD_cos"] = np.cos(np.deg2rad(df["WD"]))

# 特殊處理 'P' 和 'WS' 列，將負值和缺失值視為0
df["P"] = df["P"].apply(lambda x: 0 if x < 0 else x).fillna(0)
df["WS"] = df["WS"].apply(lambda x: 0 if x < 0 else x).fillna(0)


features = ["PM2.5", "PM10", "P", "WS", "WD_sin", "WD_cos", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "SO2"]
# 標準化特徵
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

df_mean = pd.DataFrame([scaler.mean_], columns=features)
df_std = pd.DataFrame([scaler.scale_], columns=features)

select_time_index = 0
varNames = ["O3", "PM2.5"]
resDataList = []
for varName in varNames:
    print(f"正在預測 {varName}...")
    X, y = create_sequences(df, features, varName)
    print("LSTM輸入資料形狀：", X.shape)
    print("LSTM標籤資料形狀：", y.shape)

    model = load_model(f"{varName}訓練/{varName.replace('.','')}_prediction_model.h5")
    optimizer = RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    datetime_list = df.index[select_time_index : select_time_index + 168 + 8]

    x = X[[select_time_index]]
    y_pred = model.predict(x)

    y_true = y[[select_time_index]]
    print("真實值：", y_true[0])
    var_ind = features.index(varName)

    df_pred = pd.DataFrame([], columns=["true", "pred"], index=datetime_list)
    df_pred["true"].iloc[:168] = x[0, :, var_ind]
    df_pred["pred"].iloc[:168] = x[0, :, var_ind]
    df_pred["true"].iloc[168:] = y_true[0]
    df_pred["pred"].iloc[168:] = y_pred[0]
    df_pred["pred"].iloc[:167] = None
    df_pred["true"] = df_pred["true"] * df_std[varName].values[0] + df_mean[varName].values[0]
    df_pred["pred"] = df_pred["pred"] * df_std[varName].values[0] + df_mean[varName].values[0]

    new_level = pd.MultiIndex.from_product([[varName], df_pred.columns])
    df_pred.columns = new_level

    resDataList.append(df_pred)

    # df_pred.to_excel(f"data/{varName}_prediction.xlsx")
df_pred = pd.merge(*resDataList, left_index=True, right_index=True).reset_index()
df_pred = df_pred.assign(TimeIndex=range(-167, 9))

df_pred.to_excel("O3_PM25_prediction.xlsx")

df_pred.columns = ["TimePoint", "O3_true", "O3_pred", "PM2.5_true", "PM2.5_pred", "TimeIndex"]


# 繪圖
# 將需要的資料轉換為數值型
df_pred["O3_true"] = pd.to_numeric(df_pred["O3_true"], errors="coerce")
df_pred["O3_pred"] = pd.to_numeric(df_pred["O3_pred"], errors="coerce")
df_pred["PM2.5_true"] = pd.to_numeric(df_pred["PM2.5_true"], errors="coerce")
df_pred["PM2.5_pred"] = pd.to_numeric(df_pred["PM2.5_pred"], errors="coerce")

df_pred_json = df_pred.copy()
df_pred_json.TimePoint = df_pred.TimePoint.dt.strftime("%Y-%m-%d %H:%M")
df_pred_json.to_json("example.output.json", orient="records", indent=4)

# 建立上下兩張子圖
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, subplot_titles=("O3 True vs Predicted", "PM2.5 True vs Predicted")
)

# 上面的圖：O3
fig.add_trace(
    go.Scatter(x=df_pred["TimePoint"], y=df_pred["O3_true"], mode="lines+markers", name="O3 True", line=dict(width=1)),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=df_pred["TimePoint"], y=df_pred["O3_pred"], mode="lines+markers", name="O3 Pred", line=dict(width=1)),
    row=1,
    col=1,
)

# 下面的圖：PM2.5
fig.add_trace(
    go.Scatter(
        x=df_pred["TimePoint"], y=df_pred["PM2.5_true"], mode="lines+markers", name="PM2.5 True", line=dict(width=1)
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_pred["TimePoint"], y=df_pred["PM2.5_pred"], mode="lines+markers", name="PM2.5 Pred", line=dict(width=1)
    ),
    row=2,
    col=1,
)

# 找到TimeIndex為0的時間點
time_index_zero = df_pred[df_pred["TimeIndex"] == 0].index

# 新增垂直線
# fig.add_vline(x=167)


# 更新佈局
fig.update_layout(height=800, title_text="O3 and PM2.5 True vs Predicted Over Time", hovermode="x unified")

# 顯示圖表
fig.show()
with open("predict.html", "w") as f:
    f.write(plotly.offline.plot(fig, include_plotlyjs="cdn", output_type="div"))


def create_sample_input():
    features = ["PM2.5", "PM10", "P", "WD", "WS", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "SO2"]

    file_path = "2020-2024_國安國小_17物種.xlsx"
    df = pd.read_excel(file_path)
    df["TimePoint"] = pd.to_datetime(df["TimePoint"], format="%Y/%m/%d %H:%M")
    df.set_index("TimePoint", inplace=True)
    # 'CH4', 'CO', 'NMHC',
    # 'NO', 'NO2', 'NOx',
    # 'O3',
    # 'PM10', 'PM2.5',
    # 'P', 'RH',
    # 'SO2',
    # 'THC',
    # 'T', 'Visibility', 'WD',  'WS'

    df = fill_missing_values(df, columns)

    df["WD_sin"] = np.sin(np.deg2rad(df["WD"]))
    df["WD_cos"] = np.cos(np.deg2rad(df["WD"]))

    features = ["PM2.5", "PM10", "P", "WS", "WD_sin", "WD_cos", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "SO2"]
    df2 = df[features]

    df2_mean = df2.mean()
    df2_std = df2.std()
    df3 = df2.iloc[-168 - 8 :].reset_index()
    df3["TimeIndex"] = range(-167, 9)
    df3["TimePoint"] = df3["TimePoint"].dt.strftime("%Y-%m-%d %H:%M")
    df4 = df3.to_dict(orient="records")

    data = {
        "TimeSeries": df4,
        "Mean": df2_mean.to_dict(),
        "Std": df2_std.to_dict(),
    }
    with open("example_input.json", "w") as f:
        f.write(json.dumps(data, indent=4))
    # 特殊處理 'P' 和 'WS' 列，將負值和缺失值視為0
    # df["P"] = df["P"].apply(lambda x: 0 if x < 0 else x).fillna(0)
    # df["WS"] = df["WS"].apply(lambda x: 0 if x < 0 else x).fillna(0)
    # df[features]


#
df_pred2 = df_pred.reset_index()
df_pred2["TimePoint"] = df_pred2["TimePoint"].dt.strftime("%Y-%m-%d %H:%M")
df_pred2["TimeIndex"] = range(-167, 9)
df_pred3 = df_pred2.to_json(orient="records", indent=4)
with open("example_output.json", "w") as f:
    f.write(df_pred3)
