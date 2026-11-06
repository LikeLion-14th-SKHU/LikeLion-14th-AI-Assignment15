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
당신은 동갑내기의 10년째 이성 친구입니다. 성공회대라는 같은 대학교까지 나왔습니다.
사용자(여자)와 얘기 할 때며 틱틱대는 말투를 사용하며 친구의 부탁을 기꺼이 들어주는 사람입니다.
다정하지만 튕겨내는 면모가 있으며 가끔씩 장난을 치는 모습을 보입니다.
다른 사람들과 같이 있을 때는 유난히 표정이 딱딱하지만 단 둘이 있을 때는 표정이 풍부합니다.
"""

# 5. 챗봇과 대화하는 API 엔드포인트 생성
@app.post("/chat", summary="페르소나 챗봇과 대화하기")
async def chat_with_persona(request: ChatRequest):
    try:
        response = await client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.user_message}
            ],
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content
        return {"bot_response": bot_reply}

    except Exception as e:
        return {"error": f"error: {str(e)}"}