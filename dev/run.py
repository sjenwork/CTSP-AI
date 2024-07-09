from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import RMSprop
from matplotlib import pyplot as plt




def fill_missing_values(df, columns):
    for column in columns:
        for date, group in df.groupby(df.index.date):
            # 当天有值的部分平均值（不包含0和负值）
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
                            prev_day_data = df.loc[prev_day_start:prev_day_start +
                                                   pd.Timedelta(hours=23, minutes=59, seconds=59)]
                            prev_day_values = prev_day_data[column].replace(
                                [0, -np.inf, np.inf], np.nan)
                            prev_day_values = prev_day_values[prev_day_values >= 0].dropna(
                            )
                            prev_day_non_null_count = prev_day_values.count()
                            if prev_day_non_null_count > 0:
                                prev_day_mean = prev_day_values.mean()
                                df.at[i, column] = prev_day_mean
                                break
                            prev_day_start -= pd.Timedelta(days=1)


def create_sequences(df, features, target, past_hours=168, future_hours=8):
    X, y = [], []
    for i in range(len(df) - past_hours - future_hours):
        X.append(df[features].iloc[i:i+past_hours].values)
        y.append(df[target].iloc[i+past_hours:i +
                 past_hours+future_hours].values)
    return np.array(X), np.array(y)


# 讀取Excel檔案
file_path = '2020-2024_國安國小_17物種.xlsx'
df = pd.read_excel(file_path)

df = df.iloc[:500]
# 將 'TimePoint' 列轉換為日期時間格式並設置為索引
df['TimePoint'] = pd.to_datetime(df['TimePoint'], format='%Y/%m/%d %H:%M')
df.set_index('TimePoint', inplace=True)

columns = df.columns
features = ['PM2.5', 'PM10', 'P', 'WD', 'WS', 'CH4',
            'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'SO2', 'RH']
# features = ['PM2.5', 'PM10', 'P', 'WD', 'WS', 'CH4',
#             'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3',]

# 填補缺失值
fill_missing_values(df, features)

df['WD_sin'] = np.sin(np.deg2rad(df['WD']))
df['WD_cos'] = np.cos(np.deg2rad(df['WD']))

# 特殊处理 'P' 和 'WS' 列，将负值和缺失值视为0
df['P'] = df['P'].apply(lambda x: 0 if x < 0 else x).fillna(0)
df['WS'] = df['WS'].apply(lambda x: 0 if x < 0 else x).fillna(0)



features = ['PM2.5', 'PM10', 'P', 'WS', 'WD_sin', 'WD_cos',
            'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'SO2']
# 標準化特徵
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X, y = create_sequences(df, features, 'O3')
print("LSTM輸入數據形狀：", X.shape)
print("LSTM標籤數據形狀：", y.shape)


model = load_model("O3訓練\O3_prediction_model.h5")
optimizer = RMSprop(learning_rate=0.01)
model.compile(optimizer=optimizer, loss="mean_squared_error")

select_time_index = 0
datetime_list = df.index[select_time_index : select_time_index + 168 + 8]

x = X[[select_time_index]]
y_pred = model.predict(x)

y_true = y[[select_time_index]]
var_ind = features.index("O3")

df_pred = pd.DataFrame([], columns=["true", "pred"], index=datetime_list)
df_pred["true"].iloc[:168] = x[0, :, var_ind]
df_pred["true"].iloc[168:] = y_true[0]
df_pred["pred"].iloc[168:] = y_pred[0]

df_pred.plot()
