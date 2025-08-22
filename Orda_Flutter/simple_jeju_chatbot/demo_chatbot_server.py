"""
데모데이용 개인화된 제주도 여행 챗봇 서버
FastAPI 기반으로 개인화된 챗봇 서비스 제공
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import asyncio
import uvicorn
from demo_personalized_chatbot import (
    PersonalizedJejuChatbot,
    PersonalizedUserProfile, 
    PersonalizedGraphState,
    USER_DATA
)

app = FastAPI(title="Demo Personalized Jeju Chatbot", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class ChatRequest(BaseModel):
    message: str
    session_id: str = "demo-session"
    user_name: Optional[str] = None  # 직접 이름 전달 가능

class ChatResponse(BaseModel):
    response: str
    user_profile: Dict
    available_users: Optional[list] = None

# 전역 챗봇 인스턴스 (smart_chatbot.py와 동일)
chatbot = PersonalizedJejuChatbot()

# 세션별 상태 저장
sessions: Dict[str, Dict] = {}

@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "message": "데모데이용 개인화된 제주도 여행 챗봇 서버가 실행 중입니다!",
        "loaded_users": len(USER_DATA),
        "available_users": list(USER_DATA.keys())
    }

@app.get("/users")
async def get_available_users():
    """사용 가능한 사용자 목록 반환"""
    return {
        "users": [
            {
                "name": name,
                "personality": info["personality"],
                "travel_style": info["travel_style"]
            }
            for name, info in USER_DATA.items()
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """챗봇과의 대화 처리 (smart_chatbot.py와 동일한 방식)"""
    try:
        session_id = request.session_id
        user_message = request.message.strip()
        
        # 세션 ID 설정
        if session_id:
            chatbot.session_id = session_id
        
        # 이름이 전달된 경우 메시지에 포함
        if request.user_name and request.user_name in USER_DATA:
            user_message = f"{request.user_name}: {user_message}"
        
        # 챗봇 실행 (smart_chatbot.py와 동일)
        result = await chatbot.chat(user_message)
        
        # 프로필 정보를 딕셔너리로 변환
        profile_dict = {}
        if result.get("user_profile"):
            profile = result["user_profile"]
            profile_dict = {
                "name": getattr(profile, 'name', None),
                "personality": getattr(profile, 'personality', None),
                "travel_style": getattr(profile, 'travel_style', None),
                "travel_dates": getattr(profile, 'travel_dates', None),
                "duration": getattr(profile, 'duration', None),
                "group_type": getattr(profile, 'group_type', None),
                "interests": getattr(profile, 'interests', []),
                "budget": getattr(profile, 'budget', None),
                "travel_region": getattr(profile, 'travel_region', None)
            }
        
        # 사용자 이름이 없는 경우 사용 가능한 사용자 목록 포함
        available_users = None
        if not profile_dict.get("name"):
            available_users = list(USER_DATA.keys())
        
        return ChatResponse(
            response=result["response"],
            user_profile=profile_dict,
            available_users=available_users
        )
        
    except Exception as e:
        print(f"챗봇 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"챗봇 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/set_user")
async def set_user_endpoint(request: dict):
    """사용자 직접 설정"""
    try:
        session_id = request.get("session_id", "demo-session")
        user_name = request.get("user_name")
        
        if not user_name or user_name not in USER_DATA:
            raise HTTPException(status_code=400, detail="유효하지 않은 사용자 이름입니다.")
        
        # 세션 초기화
        if session_id not in sessions:
            sessions[session_id] = {
                "conversation_history": [],
                "user_profile": PersonalizedUserProfile()
            }
        
        # 사용자 정보 설정
        user_info = USER_DATA[user_name]
        sessions[session_id]["user_profile"].name = user_name
        sessions[session_id]["user_profile"].personality = user_info["personality"]
        sessions[session_id]["user_profile"].travel_style = user_info["travel_style"]
        
        return {
            "message": f"{user_name}님으로 설정되었습니다!",
            "user_profile": sessions[session_id]["user_profile"].to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """세션 초기화"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"세션 {session_id}이 초기화되었습니다."}
    else:
        return {"message": "해당 세션이 존재하지 않습니다."}

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """세션 정보 조회"""
    if session_id not in sessions:
        return {"message": "세션이 존재하지 않습니다."}
    
    session_data = sessions[session_id]
    return {
        "session_id": session_id,
        "user_profile": session_data["user_profile"].to_dict(),
        "conversation_count": len(session_data["conversation_history"])
    }

if __name__ == "__main__":
    print("🌴 데모데이용 개인화된 제주도 여행 챗봇 서버를 시작합니다...")
    print(f"📊 로딩된 사용자 데이터: {len(USER_DATA)}명")
    print("🚀 서버 주소: http://localhost:8004")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8004,
        log_level="info"
    )