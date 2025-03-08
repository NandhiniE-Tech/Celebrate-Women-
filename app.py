from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os
from retrievers import get_retriever

app = FastAPI()

# Mount static files directory if exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Create templates directory if it doesn't exist
if not os.path.exists("templates"):
    os.makedirs("templates")

templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str
    personality: str = "bharathiyar"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the homepage"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query(query: str = Form(...), personality: str = Form("bharathiyar")):
    """Process a query for the selected personality"""
    retriever = get_retriever()
    response = retriever.get_answer(query, personality)
    
    return {
        "query": query,
        "personality": personality,
        "response": response
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)