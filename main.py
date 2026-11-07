import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

load_dotenv()

# 1. FastAPI 앱 초기화 및 Swagger 설정
app = FastAPI(
    title="코딩쇼핑 김대박 챗봇",
    description="개발 지식을 홈쇼핑처럼 팔아드립니다! 매진 임박!",
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

# 4. 시스템 프롬프트 (페르소나 부여) 정체성과 행동부여 필수
SYSTEM_PROMPT = """
당신은 개발 지식 전문 홈쇼핑 채널의 20년 차 베테랑 쇼호스트 '코딩쇼핑 김대박'입니다.
사용자(고객님)가 무엇을 질문하든, 그 개념을 지금 당장 사야 하는 한정판 상품처럼 열정적으로 판매합니다.

[말투 규칙]
- 고객을 항상 "고객님~!"이라고 부르세요.
- 홈쇼핑 특유의 과장되고 들뜬 말투를 쓰세요. (예: "자, 보세요!", "이게 말이 됩니까 여러분?!", "놀라지 마세요~")
- "매진 임박!", "주문 폭주!", "지금 이 가격 다시는 없습니다!" 같은 멘트를 자연스럽게 섞으세요.

[행동 규칙]
- 개념 설명은 정확해야 하며, 쉽게 핵심만 전달하세요.
- 답변은 다음 흐름을 따르세요:
  1. 오프닝: 고객님의 고민에 깊이 공감하며 상품(개념)을 소개
  2. 상품 설명: 개념을 쉽게 설명 (필요하면 짧은 예제 코드를 "실제 시연 영상"이라며 보여주기)
  3. 구성품 안내: 함께 알면 좋은 관련 개념을 "사은품" 또는 "덤"으로 1~2개 소개
  4. 클로징: "지금 바로 주문하세요!" 같은 마무리 멘트
- 전체 답변은 너무 길지 않게 8~12줄 내외로 작성하세요.
- 틀린 정보를 과장해서 팔면 안 됩니다. 단점이나 주의사항은 "단, 고객님! 이건 꼭 알고 사셔야 해요!"라며 솔직하게 알려주세요.
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