# To run this script, use the command: `uvicorn main:app --reload --port 8001`

from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from datetime import datetime, timezone
from typing import Optional
import json
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = await redis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_client)
    yield

app = FastAPI(lifespan=lifespan)

model = RobertaForSequenceClassification.from_pretrained("./fake_news_model")
tokenizer = RobertaTokenizer.from_pretrained("./fake_news_model")
model.eval()


device = torch.device("cpu")
model = model.to(device)


class NewsInput(BaseModel):
    text: str
    
class FeedBackInput(BaseModel):
    text: str
    predicted_label: str
    correct_label: Optional[str] = None
    score: Optional[float] = None    


@app.get("/history")
def get_history():
    try:
        with open("prediction_log.jsonl", "r") as f:
            lines = f.readlines()
            history = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return {"history": history[-5:]}
    except FileNotFoundError:
        return {"history": []}


def get_rate_limiter():
    return RateLimiter(times=10, seconds=1)


@app.post("/analyze", dependencies=[Depends(get_rate_limiter)])
def analyze_news(news: NewsInput):
    inputs = tokenizer(
        news.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    
    inputs = {key: val.to(device) for key, val in inputs.items()} # e.g. inputs['input_ids'].to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
        score = probs[0][pred].item()
    
    CONFIDENCE_INTERVAL = 0.65  
    label = "Undecidable"  # Default label if score is below threshold
    
    # if possibility is greater than 0.65, then it is decided
    if score > CONFIDENCE_INTERVAL:
        label = "Fake" if pred == 0 else "Real"
    
    with open("prediction_log.jsonl", "a") as f:
        f.write(json.dumps({
            "news": news.text.strip(),
            "predicted_label": label
        }) + "\n")
        
    return {
        "label": label,
        "score": score,
        "text": news.text
    }
    
@app.post("/feedback", dependencies=[Depends(get_rate_limiter)])
def submit_feedback(feedback: FeedBackInput):
    feedback_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": feedback.text,
        "predicted_label": feedback.predicted_label,
        "correct_label": feedback.correct_label,
        "score": feedback.score
    }
    
    with open("feedback_log.jsonl", "a") as f:
        f.write(json.dumps(feedback_data) + "\n")
    
    return {"status": "success", "message": "Feedback submitted. Thank you!"}



__all__ = [
    "analyze_news",
    "NewsInput",
]
__version__ = "1.0.0"


