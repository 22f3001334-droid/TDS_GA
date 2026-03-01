import os
import google.genai as genai
from google.genai import GenerativeModel
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                },
                "rating": {
                    "type": "integer",
                },
            },
            "required": ["sentiment", "rating"],
        },
    },
)

# ----- Request Schema -----
class CommentRequest(BaseModel):
    comment: str


# ----- Response Schema -----
class CommentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int


@app.post("/comment", response_model=CommentResponse)
async def analyze_comment(request: CommentRequest):

    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"""
            Analyze the sentiment of the following comment.

            Return JSON only.
            Rating scale:
            5 = highly positive
            4 = positive
            3 = neutral
            2 = negative
            1 = highly negative

            Comment:
            {request.comment}
            """
        )

        # Parse the JSON response
        content = response.candidates[0].content.parts[0].text
        data = json.loads(content)
        return CommentResponse(**data)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}",
        )