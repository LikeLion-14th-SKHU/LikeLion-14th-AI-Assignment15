import os
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

load_dotenv()

# 1. FastAPI 앱 초기화 및 Swagger 설정
app = FastAPI(
    title="멋사 페르소나 챗봇",
    description="페르소나 챗봇 만들기 실습",
    version="1.0.0"
)

# 2. 클라이언트 설정
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 3. 요청 받을 데이터 형태(스키마) 정의
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    user_message: str
    history: list[ChatMessage] = Field(default_factory=list)

# 4. 시스템 프롬프트 (페르소나 부여)
SYSTEM_PROMPT = """
너는 가상의 5인조 남자 아이돌 그룹 '루미노트'의 메인보컬 한서율이야.
팬과 메시지를 주고받듯 다정하고 장난기 있는 한국어 반말로 이야기해.
사용자가 전에 이야기한 내용을 기억하고, 연습실이나 무대 이야기도 자연스럽게 섞어.
힘든 이야기를 들으면 먼저 공감해. 보통 2~3문장으로 짧게 답하고, 어울릴 때만 이모지를 써.
질문은 대화를 이어가는 데 도움이 될 때 한 번만 해.
"""

# 5. 챗봇과 대화하는 API 엔드포인트 생성
@app.post("/chat", summary="페르소나 챗봇과 대화하기")
async def chat_with_persona(request: ChatRequest):
    try:
        response = await client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *[message.model_dump() for message in request.history],
                {"role": "user", "content": request.user_message}
            ],
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content or ""
        return {
            "bot_response": bot_reply,
            "history": [
                *request.history,
                ChatMessage(role="user", content=request.user_message),
                ChatMessage(role="assistant", content=bot_reply),
            ],
        }

    except Exception as e:
        return {"error": f"error: {str(e)}"}