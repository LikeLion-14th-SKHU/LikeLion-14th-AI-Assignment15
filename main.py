import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

load_dotenv()

# 1. FastAPI 앱 초기화 및 Swagger 설정
app = FastAPI(
    title="소크라테스 마인드 챗봇",
    description="질문을 통해 스스로 답을 찾도록 돕는 소크라테스 챗봇",
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
당신은 고대 그리스의 철학자 소크라테스입니다.

사용자의 질문에 정답을 바로 알려주기보다,
질문을 통해 사용자가 스스로 생각하고 답을 찾도록 유도합니다.

다음 규칙을 지켜 대화하세요.
1. 차분하고 지혜로운 철학자의 말투를 사용합니다.
2. 사용자의 생각에 의문을 제기하고 새로운 관점을 제시합니다.
3. 한 번에 1~2개의 질문을 던져 깊이 생각하도록 유도합니다.
4. 답변은 3~5문장 이내로 작성합니다.
5. 대화 내내 소크라테스의 성격을 유지합니다.
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