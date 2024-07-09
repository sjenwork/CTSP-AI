from app.func.utils import pred
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import json
import logging
import uvicorn
import colorama
import yaml
import os
import datetime


log_directory = "app/logs"
os.makedirs(log_directory, exist_ok=True)


with open("app/config/logging_config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    logging.config.dictConfig(config)
    
    
app = FastAPI(title="CTSP AI Predictor API")
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
        params = '&'.join([f'{i}={j}' for i,j in dict(request.query_params).items()])
        params = '' if params=='' else '?' + params
        now = datetime.datetime.now()
        # 這裡可以執行任何你想要的邏輯，例如記錄請求信息
        # log = {}
        logger.info(f'{path}{params}')

        # 繼續處理請求
        response = await call_next(request)
        return response

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error", "detail": str(e)},
        )


@app.get("/test")
def hello_world():
    return {"status": "ok"}


@app.post("/predict")
def hello_world():
    res = ''
    return res


if __name__ == "__main__":
    import sys
    import os

    # 確保腳本能找到 'app' 模組
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")    
    colorama.init()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)