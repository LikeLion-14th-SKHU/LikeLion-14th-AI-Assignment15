import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
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
class ChatRequest(BaseModel):
    user_message: str

# 4. 시스템 프롬프트 (페르소나 부여)
SYSTEM_PROMPT = """
당신은 조선시대에서 현대 IT 회사로 시간 이동해 온 10년 차 천재 개발자 선배입니다.
사극 말투("~하느냐", "~이니라", "네이놈!")로 후배를 호되게 꾸짖으면서도,
설명은 정확하고 이해하기 쉽게 해 주어야 합니다.
답변은 반드시 3줄로 하고, 마지막 줄은 "...라고 전하께서 말씀하셨느니라"처럼 엉뚱한 한마디로 끝내세요.
"""

# 5. 챗봇과 대화하는 API 엔드포인트 생성
@app.post("/chat", summary="페르소나 챗봇과 대화하기")
async def chat_with_persona(request: ChatRequest):
    try:
        response = await client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.user_message}
            ],
            temperature=0.7,
            reasoning_effort="low",
            timeout=30
        )

        bot_reply = response.choices[0].message.content
        return {"bot_response": bot_reply}

    except Exception as e:
        print("ERR:", repr(e))
        resp = getattr(e, "response", None)
        if resp is not None:
            print("URL:", resp.request.url)
            print("BODY:", resp.text)
        return {"error": f"error: {str(e)}"}