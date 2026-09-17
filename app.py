from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from predict import predict_news



# CREATE FASTAPI APP


app = FastAPI(
    title="NewsVerify AI",
    description="ML based Fake and Real News Detection System",
    version="1.0"
)



# REQUEST DATA FORMAT


class NewsRequest(BaseModel):
    news: str



# SERVE FRONTEND FILES


app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)



# HOME ROUTE


@app.get("/")
def home():
    return FileResponse("frontend/index.html")



# PREDICTION ROUTE


@app.post("/predict")
def predict(request: NewsRequest):

    result = predict_news(request.news)

    return {
        "prediction": result
    }