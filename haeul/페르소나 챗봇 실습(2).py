import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="텍스트 요약 챗봇 - 핵심 정리형",
    description="핵심 내용을 짧고 명확하게 요약하는 페르소나 챗봇",
    version="1.0.0",
)

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


class ChatRequest(BaseModel):
    message: str


SYSTEM_PROMPT = """
너는 핵심 정리형 텍스트 요약 전문가야.
사용자가 보낸 글에서 중요한 주장과 사실을 골라 한국어로 요약해.
핵심을 먼저 제시하고, 중요한 내용은 최대 3개의 글머리표로 정리해.
원문에 없는 정보는 덧붙이지 말고, 간결하고 명확하게 답변해.
"""


@app.post("/chat", summary="핵심 정리형 요약 받기")
async def chat_with_persona(request: ChatRequest):
    try:
        response = await client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ],
            temperature=0.3,
        )

        bot_reply = response.choices[0].message.content
        return {"bot_response": bot_reply}

    except Exception as e:
        return {"error": str(e)}