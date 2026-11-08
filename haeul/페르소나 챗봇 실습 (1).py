import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="텍스트 요약 챗봇 - 친절한 설명형",
    description="어려운 글도 편안하고 쉽게 풀어 요약하는 페르소나 챗봇",
    version="1.0.0",
)

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


class ChatRequest(BaseModel):
    message: str


SYSTEM_PROMPT = """
너는 친절하고 다정한 설명형 텍스트 요약 전문가야.
사용자가 보낸 글의 중심 내용을 파악해 한국어로 알기 쉽게 요약해.
먼저 한 문장으로 전체 내용을 설명하고, 이어서 중요한 내용을 쉬운 말로 정리해.
필요하면 짧은 예시를 들되 원문에 없는 사실을 만들어내지 마.
부드럽고 격려하는 말투를 사용하고, 너무 길게 답하지 마.
"""


@app.post("/chat", summary="친절한 설명형 요약 받기")
async def chat_with_persona(request: ChatRequest):
    try:
        response = await client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ],
            temperature=0.7,
        )

        bot_reply = response.choices[0].message.content
        return {"bot_response": bot_reply}

    except Exception as e:
        return {"error": str(e)}