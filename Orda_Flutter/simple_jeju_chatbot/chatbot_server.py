"""
제주도 여행 챗봇 FastAPI 서버
backend에서 호출 가능한 API 서버
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 기존 챗봇 시스템 import
from smart_chatbot import JejuTravelChatbot, UserProfile

# 요청/응답 모델 정의
class ChatRequest(BaseModel):
    content: str
    session_id: str
    conversation_history: List[Dict] = []
    user_profile: Dict = {}
    profile_completion: float = 0.0

class ChatResponse(BaseModel):
    response: str
    session_id: str
    needs_more_info: bool
    profile_completion: float
    follow_up_questions: List[str] = []
    user_profile: Dict
    analysis_confidence: float
    timestamp: str

# FastAPI 앱 생성
app = FastAPI(
    title="제주도 여행 챗봇 서비스",
    description="LangGraph 기반 멀티 에이전트 제주도 여행 상담 챗봇",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 챗봇 인스턴스
chatbot = None

@app.on_event("startup")
async def startup_event():
    """서버 시작시 챗봇 초기화"""
    global chatbot
    try:
        chatbot = JejuTravelChatbot()
        print("🚀 제주도 여행 챗봇 서비스 시작됨")
    except Exception as e:
        print(f"❌ 챗봇 초기화 실패: {e}")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "제주도 여행 챗봇 서비스",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chatbot_ready": chatbot is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 엔드포인트"""
    if not chatbot:
        raise HTTPException(
            status_code=503,
            detail="챗봇 서비스를 사용할 수 없습니다."
        )
    
    try:
        # UserProfile 객체 생성
        profile_data = request.user_profile
        user_profile = UserProfile(
            travel_dates=profile_data.get("travel_dates"),
            duration=profile_data.get("duration"),
            group_type=profile_data.get("group_type"),
            interests=profile_data.get("interests", []),
            budget=profile_data.get("budget"),
            travel_region=profile_data.get("travel_region")
        )
        
        # 챗봇 응답 생성
        response_data = await chatbot.process_message(
            user_message=request.content,
            conversation_history=request.conversation_history,
            user_profile=user_profile,
            session_id=request.session_id
        )
        
        # 응답 데이터 변환
        return ChatResponse(
            response=response_data["response"],
            session_id=request.session_id,
            needs_more_info=response_data.get("needs_more_info", False),
            profile_completion=response_data.get("profile_completion", 0.0),
            follow_up_questions=response_data.get("follow_up_questions", []),
            user_profile=response_data.get("user_profile", {}),
            analysis_confidence=response_data.get("analysis_confidence", 0.8),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"❌ 채팅 처리 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "chatbot_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )