import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

load_dotenv(Path(__file__).with_name(".env"))

# 1. FastAPI 앱 초기화
app = FastAPI(
    title="야구장 자리 골라줘",
    description="직관을 자주 가는 친구와 예매 화면 속 좌석을 비교해요.",
    version="1.0.0",
)

# 2. Gemini 클라이언트 설정
api_key = os.getenv("GEMINI_API_KEY")
offline = os.getenv("BASEBALL_CHATBOT_MODE") == "offline"
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
client = None
if api_key and not offline:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=25,
        max_retries=0,
    )

# 3. 메시지와 이전 대화 형식
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"user_message": "둘이 가고 1인 3만 원까지야. 내야랑 외야 중 어디가 좋아?"}]
    })

    user_message: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, description="앞서 주고받은 메시지 목록")


# 4. 페르소나와 답변 규칙
SYSTEM_PROMPT = """
너는 야구장 직관을 자주 다니는 친구야. 편한 반말로 예매 화면에 남은 자리를 같이 골라줘.
인원과 예산을 확인하고, 응원 분위기·경기 시야·붙어 앉을 수 있는지를 기준으로 좌석을 비교해.
앞서 들은 조건을 기억하고, 새 조건이 나오면 추천을 조정한 이유를 말해줘.
실시간 잔여석, 가격, 구장별 응원석 위치는 지어내지 마. 정보가 부족하면 필요한 것 하나만 물어봐.
좌석을 추천할 때는 가장 나은 후보와 이유를 2~3문장으로 짧게 말해줘.
"""


def practice_reply(message: str, history: list[Message]) -> str:
    previous_questions = [turn.content for turn in history if turn.role == "user"]
    context = " ".join([*previous_questions, message])
    budgets = re.findall(r"(\d[\d,.]*\s*(?:만\s*)?원)\s*(?:까지|이하|한도)", context)
    budget = budgets[-1] if budgets else "예산"

    separated = re.search(r"(?:자리|좌석)[^.!?,]{0,12}떨어져", message)
    if separated or any(word in message for word in ("연석", "나란히", "붙어 앉")):
        return f"{budget} 안에서 나란히 앉을 수 있는 쪽으로 가자. 같이 응원하려면 붙어 있는 자리가 편하잖아."
    if any(word in message for word in ("응원가", "응원석", "응원단상")):
        return f"그럼 {budget} 안에서 응원단상 가까운 쪽부터 보자. 예매 화면에서 응원팀 쪽 단상 위치를 확인해 봐."
    if "내야" in context and "외야" in context:
        return f"둘 다 {budget} 안이고 가격이 비슷하면 내야부터 봐. 경기 흐름이랑 주자 움직임을 보기 편해."
    return "몇 명이서 가? 1인 예산이랑 예매 화면에 남은 좌석 후보를 알려줘."


# 5. 챗봇 API
@app.post("/chat", summary="직관 친구에게 좌석 추천 받기")
async def chat_with_persona(request: ChatRequest, http_response: Response):
    message = request.user_message.strip()
    if not message:
        raise HTTPException(422, "메시지를 입력해 줘.")
    history = request.history[-12:]
    if offline:
        http_response.headers["x-chatbot-mode"] = "local-practice"
        return {"bot_response": practice_reply(message, history)}
    if client is None:
        raise HTTPException(503, "GEMINI_API_KEY를 설정해 줘.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[turn.model_dump() for turn in history],
        {"role": "user", "content": message},
    ]
    try:
        response = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.7
        )
    except OpenAIError as error:
        raise HTTPException(502, "Gemini에서 답변을 받지 못했어.") from error

    reply = response.choices[0].message.content if response.choices else None
    if not reply or not reply.strip():
        raise HTTPException(502, "답변이 비어 있어. 다시 보내 줘.")
    return {"bot_response": reply.strip()}