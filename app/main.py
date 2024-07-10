from app.func.utils import pred
from fastapi import FastAPI, Request, Query, HTTPException, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from app.models.prediction import PredictorData
from datetime import datetime, timedelta
import json
import logging
import yaml
import os
import datetime
import requests
from pprint import pprint

log_directory = "app/logs"
os.makedirs(log_directory, exist_ok=True)


with open("app/config/logging_config.yaml", "r") as file:
    config = yaml.safe_load(file)
    logging.config.dictConfig(config)


app = FastAPI(title="CTSP AI Predictor API")
router = APIRouter(prefix="/ai")


logger = logging.getLogger("prod")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    yield
    print("Shutting down...")


# app.lifespan(lifespan)
app.router.lifespan_context = lifespan


@app.middleware("http")
async def middleware(request: Request, call_next):
    try:
        # 獲取路由路徑和參數
        path = request.url.path
        params = "&".join([f"{i}={j}" for i, j in dict(request.query_params).items()])
        params = "" if params == "" else "?" + params
        now = datetime.datetime.now()
        # 這裡可以執行任何你想要的邏輯，例如記錄請求信息
        # log = {}
        logger.info(f"{path}{params}")

        # 繼續處理請求
        response = await call_next(request)
        return response

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error", "detail": str(e)},
        )


@router.get("/api/test")
def test():
    return {"status": "ok"}


@router.post("/api/predict")
async def receive_data(data: PredictorData):
    res = pred(data.model_dump())
    return res


# 以下為 templates 的部分
templates = Jinja2Templates(directory="app/templates")


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return HTMLResponse(
        content=f"""
        <html>
            <head>
                <title>{exc.status_code} Error</title>
            </head>
            <body>
                <h1>{exc.status_code} Error</h1>
                <p>{exc.detail}</p>
            </body>
        </html>
        """,
        status_code=exc.status_code,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return HTMLResponse(
        content="""
        <html>
            <head>
                <title>422 Unprocessable Entity</title>
            </head>
            <body>
                <h1>422 Unprocessable Entity</h1>
                <p>Validation error occurred. Please check your parameters.</p>
            </body>
        </html>
        """,
        status_code=422,
    )


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    print(request.url.path)
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/plot/", response_class=HTMLResponse)
async def get_plot(request: Request, station: str = Query(...), datetime: str = Query(...)):

    # 取得數據
    data = get_data(station, datetime)
    # 預測
    res = pred(data)
    df = pd.DataFrame(res)
    pprint(df.iloc[-20:])

    # 創建子圖
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, subplot_titles=("O3 Levels", "PM2.5 Levels"), vertical_spacing=0.1
    )

    # 添加 O3 的真實值和預測值
    fig.add_trace(
        go.Scatter(
            x=df["TimePoint"],
            y=df["O3_true"],
            mode="lines+markers",
            name="O3 觀測",
            line=dict(width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Value: %{y}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["TimePoint"],
            y=df["O3_pred"],
            mode="lines+markers",
            name="O3 預測",
            line=dict(width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Value: %{y}",
        ),
        row=1,
        col=1,
    )

    # 添加 PM2.5 的真實值和預測值
    fig.add_trace(
        go.Scatter(
            x=df["TimePoint"],
            y=df["PM25_true"],
            mode="lines+markers",
            name="PM2.5 觀測",
            line=dict(width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Value: %{y}",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["TimePoint"],
            y=df["PM25_pred"],
            mode="lines+markers",
            name="PM2.5 預測",
            line=dict(width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Value: %{y}",
        ),
        row=2,
        col=1,
    )

    # 更新佈局
    fig.update_layout(
        font=dict(size=18),  # 調整這個數字來改變字體大小
        height=600,
        autosize=True,
        legend=dict(x=0.5, y=1.15, xanchor="center", yanchor="top", orientation="h"),  # 調整圖例的位置
        margin=dict(t=80),  # 增加頂部邊距，避免標題和圖例擠在一起
    )
    # 轉換為HTML
    graph_html = fig.to_html(full_html=False)

    return templates.TemplateResponse(
        "index.html", {"request": request, "graph_html": graph_html, "station": station, "datetime": datetime}
    )


def get_data(station: str, datetime: str):
    # 這裡可以根據站點和日期時間查詢數據庫

    data_url = "https://raw.githubusercontent.com/sjenwork/CTSP-AI/main/dev/ctsp/example_input.json"
    res = requests.get(data_url)
    if res.ok:
        res_data = res.json()
        data = pd.DataFrame(res_data["TimeSeries"])
        data.loc[:, "TimePoint"] = create_time(datetime)
        res_data["TimeSeries"] = data.to_dict(orient="records")
        return res_data
    else:
        return None


def create_time(select_time):
    time0 = pd.to_datetime(select_time)
    start_time = time0 - timedelta(hours=24 * 7 - 1)
    end_time = time0 + timedelta(hours=8)
    time_points = pd.date_range(start=start_time, end=end_time, freq="h")
    return time_points


app.include_router(router)
if __name__ == "__main__":
    import sys
    import os
    import colorama
    import uvicorn

    # 確保腳本能找到 'app' 模組
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
    colorama.init()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
