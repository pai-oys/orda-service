"""
🌴 LangGraph 기반 제주도 여행 멀티 에이전트 시스템
- 플래너 에이전트: 사용자 프로필 수집
- 숙박 에이전트: 호텔/펜션 정보 검색
- 관광 에이전트: 관광지 정보 검색  
- 음식 에이전트: 맛집 정보 검색
- 행사 에이전트: 이벤트 정보 검색
- 응답 생성 에이전트: 최종 일정 추천
"""

import asyncio
import httpx
import json
from typing import Dict, List, Optional, TypedDict
from dataclasses import dataclass, asdict
from datetime import datetime
from langchain_upstage import ChatUpstage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# 환경변수에서 API 키 가져오기
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

@dataclass
class UserProfile:
    """사용자 프로필 정보"""
    travel_dates: Optional[str] = None
    duration: Optional[str] = None  
    group_type: Optional[str] = None
    interests: List[str] = None
    budget: Optional[str] = None
    travel_region: Optional[str] = None
    
    def __post_init__(self):
        if self.interests is None:
            self.interests = []
    
    def to_dict(self):
        return asdict(self)
    
    def get_summary(self) -> str:
        """프로필 요약 텍스트 생성"""
        summary_parts = []
        if self.travel_dates:
            summary_parts.append(f"날짜: {self.travel_dates}")
        if self.duration:
            summary_parts.append(f"기간: {self.duration}")
        if self.group_type:
            summary_parts.append(f"여행 유형: {self.group_type}")
        if self.interests:
            summary_parts.append(f"관심사: {', '.join(self.interests)}")
        if self.budget:
            summary_parts.append(f"예산: {self.budget}")
        if self.travel_region:
            summary_parts.append(f"여행지역: {self.travel_region}")
        
        return " | ".join(summary_parts) if summary_parts else "정보 없음"

# LangGraph State 정의
class GraphState(TypedDict):
    """그래프 상태"""
    user_message: str
    conversation_history: List[Dict]
    user_profile: UserProfile
    hotel_results: List[Dict]
    travel_results: List[Dict] 
    food_results: List[Dict]
    event_results: List[Dict]
    final_response: str
    profile_ready: bool

# 공용 LLM 인스턴스들 (각 에이전트별 독립 LLM)
profile_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
hotel_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
travel_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
food_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
event_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
response_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")

# 벡터 DB 접근 URL
RAG_URL = "http://localhost:8002/chat"

# 프로필 수집 노드
async def profile_collector_node(state: GraphState) -> GraphState:
    """사용자 프로필 수집 및 업데이트"""
    user_message = state["user_message"]
    conversation_history = state.get("conversation_history", [])
    current_profile = state.get("user_profile", UserProfile())
    
    # 대화 기록에 사용자 메시지 추가
    conversation_history.append({
        "role": "user",
        "message": user_message,
        "timestamp": datetime.now().isoformat()
    })
    
    # 프로필 정보 추출
    profile_info = await extract_profile_info(user_message, current_profile)
    
    # 프로필 업데이트
    updated_profile = update_profile(current_profile, profile_info)
    
    # 프로필이 충분한지 확인
    profile_ready = is_profile_sufficient(updated_profile)
    
    if not profile_ready:
        # 추가 정보 수집 응답 생성
        response = await generate_info_collection_response(updated_profile, user_message)
        conversation_history.append({
            "role": "assistant", 
            "message": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            **state,
            "conversation_history": conversation_history,
            "user_profile": updated_profile,
            "final_response": response,
            "profile_ready": False
        }
    
    return {
        **state,
        "conversation_history": conversation_history,
        "user_profile": updated_profile,
        "profile_ready": True
    }

# 숙박 에이전트 노드
async def hotel_agent_node(state: GraphState) -> GraphState:
    """숙박 장소 검색 에이전트"""
    user_profile = state["user_profile"]
    
    # 가이드라인 프롬프트 사용
    prompt = f"""너는 사용자의 프로필을 참고하여 벡터 DB에서 원하는 정보를 검색하기 위한 쿼리를 만드는 전문가야. 사용자의 프로필을 바탕으로 사용자가 원하는 숙박 장소를 벡터 DB에서 검색하기 위해 사용될 검색 쿼리를 한줄로 만들어줘.

사용자 프로필: {user_profile.get_summary()}

검색 쿼리만 출력해줘:"""
    
    try:
        # 쿼리 생성
        response = await hotel_llm.ainvoke(prompt)
        search_query = response.content.strip()
        
        # 벡터 DB 검색
        hotel_results = await search_vector_db(search_query, "hotel")
        
        return {
            **state,
            "hotel_results": hotel_results
        }
        
    except Exception as e:
        print(f"❌ 숙박 에이전트 오류: {e}")
        return {
            **state,
            "hotel_results": []
        }

# 관광 에이전트 노드  
async def travel_agent_node(state: GraphState) -> GraphState:
    """관광 장소 검색 에이전트"""
    user_profile = state["user_profile"]
    
    prompt = f"""너는 사용자의 프로필을 참고하여 벡터 DB에서 원하는 정보를 검색하기 위한 쿼리를 만드는 전문가야. 사용자의 프로필을 바탕으로 사용자가 원하는 관광 장소를 벡터 DB에서 검색하기 위해 사용될 검색 쿼리를 한줄로 만들어줘.

사용자 프로필: {user_profile.get_summary()}

검색 쿼리만 출력해줘:"""
    
    try:
        response = await travel_llm.ainvoke(prompt)
        search_query = response.content.strip()
        
        travel_results = await search_vector_db(search_query, "travel")
        
        return {
            **state,
            "travel_results": travel_results
        }
        
    except Exception as e:
        print(f"❌ 관광 에이전트 오류: {e}")
        return {
            **state,
            "travel_results": []
        }

# 음식 에이전트 노드
async def food_agent_node(state: GraphState) -> GraphState:
    """식당 검색 에이전트"""
    user_profile = state["user_profile"]
    
    prompt = f"""너는 사용자의 프로필을 참고하여 벡터 DB에서 원하는 정보를 검색하기 위한 쿼리를 만드는 전문가야. 사용자의 프로필을 바탕으로 사용자가 원하는 식당을 벡터 DB에서 검색하기 위해 사용될 검색 쿼리를 한줄로 만들어줘.

사용자 프로필: {user_profile.get_summary()}

검색 쿼리만 출력해줘:"""
    
    try:
        response = await food_llm.ainvoke(prompt)
        search_query = response.content.strip()
        
        food_results = await search_vector_db(search_query, "food")
        
        return {
            **state,
            "food_results": food_results
        }
        
    except Exception as e:
        print(f"❌ 음식 에이전트 오류: {e}")
        return {
            **state,
            "food_results": []
        }

# 행사 에이전트 노드
async def event_agent_node(state: GraphState) -> GraphState:
    """행사 검색 에이전트"""
    user_profile = state["user_profile"]
    
    prompt = f"""너는 사용자의 프로필을 참고하여 벡터 DB에서 원하는 정보를 검색하기 위한 쿼리를 만드는 전문가야. 사용자의 프로필을 바탕으로 사용자의 상황에 맞는 행사를 추천를 벡터 DB에서 검색하기 위해 사용될 쿼리를 한줄로 만들어줘.

사용자 프로필: {user_profile.get_summary()}

검색 쿼리만 출력해줘:"""
    
    try:
        response = await event_llm.ainvoke(prompt)
        search_query = response.content.strip()
        
        event_results = await search_vector_db(search_query, "event")
        
        return {
            **state,
            "event_results": event_results
        }
        
    except Exception as e:
        print(f"❌ 행사 에이전트 오류: {e}")
        return {
            **state,
            "event_results": []
        }

# 응답 생성 노드
async def response_generator_node(state: GraphState) -> GraphState:
    """최종 응답 생성 에이전트"""
    user_profile = state["user_profile"]
    hotel_results = state.get("hotel_results", [])
    travel_results = state.get("travel_results", [])
    food_results = state.get("food_results", [])
    event_results = state.get("event_results", [])
    conversation_history = state.get("conversation_history", [])
    
    prompt = f"""다음 정보를 종합하여 사용자에게 제주도 여행 일정을 추천해주세요.

**사용자 프로필:**
{user_profile.get_summary()}

**숙박 정보 (hotel):**
{json.dumps(hotel_results[:3], ensure_ascii=False, indent=2)}

**관광 정보 (travel):**
{json.dumps(travel_results[:5], ensure_ascii=False, indent=2)}

**음식 정보 (food):**
{json.dumps(food_results[:5], ensure_ascii=False, indent=2)}

**행사 정보 (event):**
{json.dumps(event_results[:3], ensure_ascii=False, indent=2)}

위 정보를 바탕으로 사용자에게 상세하고 실용적인 제주도 여행 일정을 추천해주세요. 
- 시간대별 일정 포함
- 지리적 효율성 고려
- 구체적인 장소 정보 제공
- 현실적이고 실행 가능한 일정"""
    
    try:
        response = await response_llm.ainvoke(prompt)
        final_response = response.content.strip()
        
        # 대화 기록에 응답 추가
        conversation_history.append({
            "role": "assistant",
            "message": final_response, 
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            **state,
            "final_response": final_response,
            "conversation_history": conversation_history
        }
        
    except Exception as e:
        print(f"❌ 응답 생성 오류: {e}")
        return {
            **state,
            "final_response": "죄송합니다. 일정 생성 중 오류가 발생했습니다.",
            "conversation_history": conversation_history
        }

# 유틸리티 함수들
async def extract_profile_info(message: str, current_profile: UserProfile) -> Dict:
    """메시지에서 프로필 정보 추출"""
    prompt = f"""다음 사용자 메시지에서 제주도 여행 관련 정보를 추출해주세요.

사용자 메시지: {message}

현재 프로필: {current_profile.get_summary()}

다음 정보를 JSON 형태로 추출해주세요 (없으면 null):

{{
    "travel_dates": "여행 날짜 (예: 8월 1일-3일, 다음주 금요일부터 등)",
    "duration": "여행 기간 (예: 2박3일, 3일, 1주일 등)", 
    "group_type": "여행 유형 (예: 커플, 가족, 친구, 혼자 등)",
    "interests": ["관심사 배열 (예: 액티비티, 맛집, 힐링, 사진촬영 등)"],
    "budget": "예산 정보",
    "travel_region": "여행 지역 (제주시, 서귀포 등)"
}}

명시적으로 언급된 정보만 추출하고, 애매한 표현은 null로 처리해주세요."""

    try:
        response = await profile_llm.ainvoke(prompt)
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        return json.loads(content)
        
    except Exception as e:
        print(f"❌ 프로필 추출 오류: {e}")
        return {}

def update_profile(current_profile: UserProfile, profile_info: Dict) -> UserProfile:
    """프로필 업데이트"""
    if profile_info.get("travel_dates"):
        current_profile.travel_dates = profile_info["travel_dates"]
    if profile_info.get("duration"):
        current_profile.duration = profile_info["duration"]
    if profile_info.get("group_type"):
        current_profile.group_type = profile_info["group_type"]
    if profile_info.get("interests"):
        new_interests = profile_info["interests"]
        for interest in new_interests:
            if interest not in current_profile.interests:
                current_profile.interests.append(interest)
    if profile_info.get("budget"):
        current_profile.budget = profile_info["budget"]
    if profile_info.get("travel_region"):
        current_profile.travel_region = profile_info["travel_region"]
        
    return current_profile

def is_profile_sufficient(profile: UserProfile) -> bool:
    """프로필이 충분한지 확인"""
    # 기본적인 정보가 있으면 충분하다고 판단
    return bool(
        profile.duration and 
        profile.group_type and 
        profile.interests
    )

async def generate_info_collection_response(profile: UserProfile, user_message: str) -> str:
    """정보 수집 응답 생성"""
    prompt = f"""제주도 여행 상담사로서 사용자와 자연스럽게 대화하면서 필요한 정보를 수집해주세요.

현재 수집된 정보: {profile.get_summary()}
사용자 메시지: {user_message}

친근하고 자연스러운 톤으로 추가 정보를 요청하거나 현재 정보로 추천을 시작할 수 있음을 알려주세요."""

    try:
        response = await profile_llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"❌ 정보 수집 응답 생성 오류: {e}")
        return "제주도 여행에 대해 더 자세히 알려주시면 더 좋은 추천을 드릴 수 있어요! 😊"

async def search_vector_db(query: str, category: str = "") -> List[Dict]:
    """벡터 DB 검색"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                RAG_URL,
                json={"query": query}
            )
            
            if response.status_code == 200:
                result = response.json()
                sources = result.get("sources", [])
                print(f"🔍 {category} 검색 - 쿼리: {query}, 결과: {len(sources)}개")
                return sources[:10]  # 최대 10개 결과만 반환
            else:
                print(f"❌ 벡터 DB 검색 실패 - 상태코드: {response.status_code}")
                return []
                
    except Exception as e:
        print(f"❌ 벡터 DB 검색 오류: {e}")
        return []

# 조건부 라우팅 함수
def should_continue_to_agents(state: GraphState) -> str:
    """프로필이 준비되었는지 확인하여 다음 단계 결정"""
    if state.get("profile_ready", False):
        return "agents"
    else:
        return "end"

# LangGraph 설정
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("profile_collector", profile_collector_node)
workflow.add_node("hotel_agent", hotel_agent_node)
workflow.add_node("travel_agent", travel_agent_node)
workflow.add_node("food_agent", food_agent_node)
workflow.add_node("event_agent", event_agent_node)
workflow.add_node("response_generator", response_generator_node)

# 시작점 설정
workflow.set_entry_point("profile_collector")

# 조건부 엣지 설정
workflow.add_conditional_edges(
    "profile_collector",
    should_continue_to_agents,
    {
        "agents": ["hotel_agent", "travel_agent", "food_agent", "event_agent"],
        "end": END
    }
)

# 에이전트들이 완료되면 응답 생성기로
workflow.add_edge("hotel_agent", "response_generator") 
workflow.add_edge("travel_agent", "response_generator")
workflow.add_edge("food_agent", "response_generator")
workflow.add_edge("event_agent", "response_generator")

# 응답 생성 후 종료
workflow.add_edge("response_generator", END)

# 메모리 설정 및 컴파일
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# 메인 챗봇 클래스
class SmartJejuChatbot:
    """LangGraph 기반 멀티 에이전트 제주도 여행 챗봇"""
    
    def __init__(self):
        self.graph = graph
        self.session_id = "default"
    
    async def chat(self, user_message: str) -> str:
        """사용자와 채팅"""
        initial_state = {
            "user_message": user_message,
            "conversation_history": [],
            "user_profile": UserProfile(),
            "hotel_results": [],
            "travel_results": [],
            "food_results": [],
            "event_results": [],
            "final_response": "",
            "profile_ready": False
        }
        
        try:
            # 그래프 실행
            config = {"configurable": {"thread_id": self.session_id}}
            result = await self.graph.ainvoke(initial_state, config)
            
            return result.get("final_response", "죄송합니다. 응답을 생성할 수 없습니다.")
            
        except Exception as e:
            print(f"❌ 챗봇 실행 오류: {e}")
            return "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요."

# FastAPI 서버 설정
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="🌴 LangGraph 제주도 멀티 에이전트 챗봇")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 챗봇 인스턴스
chatbot = SmartJejuChatbot()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 엔드포인트"""
    
    try:
        if request.session_id:
            chatbot.session_id = request.session_id
            
        response = await chatbot.chat(request.message)
        
        return ChatResponse(
            response=response,
            session_id=request.session_id or "default"
        )
        
    except Exception as e:
        print(f"❌ 채팅 오류: {e}")
        return ChatResponse(
            response="죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
            session_id=request.session_id or "default"
        )

@app.get("/")
async def root():
    return {"message": "🌴 LangGraph 기반 제주도 멀티 에이전트 챗봇 API"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 LangGraph 제주도 멀티 에이전트 챗봇 시작!")
    print("📍 서버: http://localhost:8003")
    uvicorn.run(app, host="0.0.0.0", port=8003)