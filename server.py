from fastapi import FastAPI
from serving.handler import Handler, Request, Response

app = FastAPI()
handler = Handler()


@app.post("/model/")
async def inference(request: Request) -> Response:
    response = handler(request)
    return response
