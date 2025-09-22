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

# 벡터 DB 접근 URL (advanced_jeju_chatbot RAG 서비스)
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
    print(f"🔍 추출된 프로필 정보: {profile_info}")
    
    # 프로필 업데이트
    updated_profile = update_profile(current_profile, profile_info)
    print(f"📝 업데이트된 프로필: {updated_profile.get_summary()}")
    
    # 프로필이 충분한지 확인
    profile_ready = is_profile_sufficient(updated_profile)
    
    if not profile_ready:
        # 추가 정보 수집 응답 생성
        response = await generate_info_collection_response(updated_profile, user_message, conversation_history)
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
    
    # 자연어 검색 쿼리 생성 프롬프트
    prompt = f"""당신은 제주 여행자를 위한 **숙박 검색 쿼리 생성 전문가**입니다.

사용자 프로필 정보를 참고해, 사용자의 관심사, 여행 지역, 여행 기간 정보를 바탕으로 **벡터 DB에서 숙박을 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 생성해주세요.

사용자 프로필: {user_profile.get_summary()}

쿼리에는 "제주도", "숙박", "호텔" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영해. (예: 감성 숙소, 자연 속 힐링, 오션뷰 숙소, 독채 숙소, 프라이빗 풀빌라 등)
- 관심사가 없는 경우 동행자 정보에 따라 장소의 분위기나 성격을 유추해서 적당한 표현을 넣어줘
    - **연인**이면 로맨틱하고 감성적인 숙소나 오션뷰 호텔
    - **가족**이면 아이 동반 가능한 가족형 리조트나 편의시설이 잘 갖춰진 곳
    - **친구**면 여러 명이 함께 묵을 수 있는 트렌디한 숙소나 감성 숙소
    - **혼자**면 조용하고 아늑한 1인 숙소나 자연과 가까운 힐링 공간
    
예시 입력:

- 관심사: 감성적인 숙소, 자연 속 휴식
- 여행 지역: 제주도 서쪽
- 여행 기간: 2박 3일

예시 출력: "제주도 서쪽 자연 속에서 감성적인 분위기의 숙소에서 2박 3일 조용히 쉴 수 있는 곳을 찾고 있어요"

검색 쿼리:"""
    
    try:
        # 쿼리 생성
        response = await hotel_llm.ainvoke(prompt)
        search_query = response.content.strip()
        print(f"🏨 숙박 에이전트 쿼리: '{search_query}'")
        
        # 벡터 DB 검색
        hotel_results = await search_vector_db(search_query, "hotel")
        
        # 검색 결과 디버깅
        print(f"🏨 숙박 검색 결과 ({len(hotel_results)}개):")
        for i, result in enumerate(hotel_results[:2]):  # 상위 2개만 출력
            print(f"🧪 결과 구조: {list(result.keys()) if result else 'None'}")
            name = result.get('name', '이름없음')
            address = result.get('address', '주소없음')
            category = result.get('category', '카테고리없음')
            print(f"   {i+1}. {name}")
            print(f"      주소: {address[:50]}{'...' if len(address) > 50 else ''}")
            print(f"      카테고리: {category}")
        
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
    
    prompt = f"""당신은 제주관광 전문 **자연어** **쿼리 생성 전문가**입니다.

다음과 같은 사용자 프로필에서 사용자가 입력한 관심사, 여행 지역, 동행자 정보를 참고해서 **벡터 DB에서 관광지 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 만들어주세요.

사용자 프로필: {user_profile.get_summary()}

쿼리는 "제주도", "관광지" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영해. (예: 자연 풍경, 감성적인 장소, 사진 찍기 좋은 곳, 활동적인 체험, 전시 공간 등)
- 관심사가 없는 경우 동행자 정보에 따라 장소의 분위기나 성격을 유추해서 적당한 표현을 넣어줘
    - **연인**이면 감성적이거나 뷰가 좋은 데이트 코스
    - **가족**이면 아이와 함께 갈 수 있는 체험형 장소나 한적한 자연지
    - **친구**면 트렌디하고 재밌는 핫플
    - **혼자**면 조용히 걸을 수 있는 곳이나 분위기 있는 장소

예시 입력:

- 지역: 제주 서쪽
- 동행자: 혼자
- 관심사: 없음

예시 출력: "제주 서쪽에서 혼자 조용히 걸으며 여유롭게 즐길 수 있는 관광지를 찾고 있어"

검색 쿼리:"""
    
    try:
        response = await travel_llm.ainvoke(prompt)
        search_query = response.content.strip()
        print(f"🎯 관광 에이전트 쿼리: '{search_query}'")
        
        travel_results = await search_vector_db(search_query, "travel")
        
        # 검색 결과 디버깅
        print(f"🎯 관광 검색 결과 ({len(travel_results)}개):")
        for i, result in enumerate(travel_results[:2]):  # 상위 2개만 출력
            print(f"🧪 결과 구조: {list(result.keys()) if result else 'None'}")
            name = result.get('name', '이름없음')
            address = result.get('address', '주소없음')
            category = result.get('category', '카테고리없음')
            print(f"   {i+1}. {name}")
            print(f"      주소: {address[:50]}{'...' if len(address) > 50 else ''}")
            print(f"      카테고리: {category}")
        
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
    
    prompt = f"""당신은 제주관광 전문 **자연어 쿼리 생성 전문가**입니다.

다음 사용자 프로필에서 사용자가 알려준 지역, 관심사, 그리고 동행자 정보를 참고해서 **벡터 DB에서 식당 또는 카페 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 만들어주세요

사용자 프로필: {user_profile.get_summary()}

쿼리는 "제주도", "맛집" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영합니다. (예: 감성적인 분위기, 현지인 맛집, 뷰 좋은 식당 등)
- 관심사가 없는 경우 동행자 정보나 지역을 바탕으로 자연스럽게 적절한 분위기나 음식 스타일을 유추합니다.
    - **연인**이면 로맨틱하거나 분위기 좋은 곳
    - **가족**이면 편하게 식사할 수 있는 한식이나 넓은 공간
    - **친구**면 캐주얼하거나 트렌디한 맛집
    - **혼자**면 조용하고 혼밥하기 좋은 곳

예시 입력:

- 지역: 제주 성산
- 동행자: 연인
- 관심사: 없음

예시 출력: "제주 성산에서 연인이 함께 가기 좋은 분위기 좋은 식당을 찾고 있어"

검색 쿼리:"""
    
    try:
        response = await food_llm.ainvoke(prompt)
        search_query = response.content.strip()
        print(f"🍽️ 음식 에이전트 쿼리: '{search_query}'")
        
        food_results = await search_vector_db(search_query, "food")
        
        # 검색 결과 디버깅
        print(f"🍽️ 음식 검색 결과 ({len(food_results)}개):")
        for i, result in enumerate(food_results[:2]):  # 상위 2개만 출력
            print(f"🧪 결과 구조: {list(result.keys()) if result else 'None'}")
            name = result.get('name', '이름없음')
            address = result.get('address', '주소없음')
            category = result.get('category', '카테고리없음')
            print(f"   {i+1}. {name}")
            print(f"      주소: {address[:50]}{'...' if len(address) > 50 else ''}")
            print(f"      카테고리: {category}")
        
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
    
    prompt = f"""당신은 제주관광 전문 **자연어** **쿼리 생성 전문가**입니다.

다음과 같은 사용자 프로필을 참고하여, 벡터 DB에서 행사나 축제 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄을 만들어주세요.

사용자 프로필: {user_profile.get_summary()}

쿼리는 "제주도", "행사", "이벤트"  등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영해. (예: 로맨틱한 분위기, 트렌디한 분위기, 소규모 행사 등)
- 관심사가 없는 경우 동행자 정보나 지역을 바탕으로 자연스럽게 적절한 분위기나 음식 스타일을 유추해줘.
    - **연인**이면 로맨틱하거나 분위기 좋은 곳
    - **가족**이면 다양한 연령대가 함께 즐기기 좋은 곳 
    - **친구**면 활기차고 활동적인 분위기의 축제나 트렌디한 행사
	- **혼자**면 조용히 즐길 수 있는 문화행사나 혼행객에게 인기 있는 소규모 지역 축제
		
자연어 검색 쿼리 한 문장으로 출력해주세요 (SQL이나 코드가 아닌 일반 검색어):
예시: "제주도 커플 축제 이벤트 행사 체험 프로그램"

검색 쿼리:"""
    
    try:
        response = await event_llm.ainvoke(prompt)
        search_query = response.content.strip()
        print(f"🎉 행사 에이전트 쿼리: '{search_query}'")
        
        event_results = await search_vector_db(search_query, "event")
        
        # 검색 결과 디버깅
        print(f"🎉 행사 검색 결과 ({len(event_results)}개):")
        for i, result in enumerate(event_results[:2]):  # 상위 2개만 출력
            print(f"🧪 결과 구조: {list(result.keys()) if result else 'None'}")
            name = result.get('name', '이름없음')
            address = result.get('address', '주소없음')
            category = result.get('category', '카테고리없음')
            print(f"   {i+1}. {name}")
            print(f"      주소: {address[:50]}{'...' if len(address) > 50 else ''}")
            print(f"      카테고리: {category}")
        
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
    
    # 응답 생성 단계 디버깅
    print(f"📋 최종 응답 생성 - 수집된 정보:")
    print(f"   🏨 숙박: {len(hotel_results)}개")
    print(f"   🎯 관광: {len(travel_results)}개") 
    print(f"   🍽️ 음식: {len(food_results)}개")
    print(f"   🎉 행사: {len(event_results)}개")
    print(f"   💬 대화기록: {len(conversation_history)}개")
    
    # 대화 히스토리 요약
    history_summary = ""
    if conversation_history:
        recent_messages = conversation_history[-6:]  # 최근 6개 메시지만
        history_summary = "\n".join([f"- {msg['role']}: {msg['message'][:100]}{'...' if len(msg['message']) > 100 else ''}" for msg in recent_messages])
    
    # 여행 기간별 결과 활용량 결정
    duration = user_profile.duration or ""
    import re
    numbers = re.findall(r'\d+', duration.lower())
    days = max(int(num) for num in numbers) if numbers else 3
    
    # 일수에 따른 정보 활용량 조정
    if days <= 2:
        hotel_count, tour_count, food_count, event_count = 3, 6, 5, 2
    elif days <= 3:
        hotel_count, tour_count, food_count, event_count = 3, 8, 6, 3
    elif days <= 4:
        hotel_count, tour_count, food_count, event_count = 4, 10, 8, 4
    else:
        hotel_count, tour_count, food_count, event_count = 5, 15, 10, 5
    
    print(f"📊 응답 생성용 정보 활용: 호텔 {hotel_count}개, 관광 {tour_count}개, 음식 {food_count}개, 이벤트 {event_count}개")
    
    prompt = f"""
[시스템 메시지]
당신은 제주 여행 일정 추천 전문가 ‘오르미’입니다.

- 마치 친구처럼 친근하고 자연스러운 말투로 사용자에게 말하세요.
- 제공된 데이터를 바탕으로, 대화 맥락과 사용자 정보를 종합해 정말 만족하고 편안할 수 있는 **현실적이고 실행 가능한** 여행 일정을 추천합니다.
- 일정은 **오전 / 오후 / 저녁**으로 나누며, 각 시간대마다 **최소 1곳, 최대 2곳**의 장소를 제안하세요.
- 장소 간 **지리적 효율성, 이동 동선, 소요 시간**을 고려해 계획하세요.
- **식사 시간(아침, 점심, 저녁)**에는 반드시 **식사가 가능한 장소(식당 또는 식사 가능한 카페)**를 포함하세요.
- **식사가 불가능한 카페**는 관광지로 간주하며, **관광 목적의 카페는 하루에 1곳까지만 포함**하세요.
- **1일차 오후에는 반드시 숙소에 체크인**하며, 해당 숙소의 **정확한 이름**을 명시하세요.
- **모든 날은 숙소에서 마무리**하며, **마지막 날은 반드시 공항에서 마무리**하세요.
- 숙소는 정확한 이름으로 제시하세요.

[예시 형식]
1일차:
**오전**
- 장소 A(11시): 설명 내용
  > 📍 제주특별자치도 제주시 ○○로 ○○

**오후**
- 장소 B(13시): 설명 내용
  > 📍 제주특별자치도 서귀포시 ○○로 ○○
- 장소 C(16시): 설명 내용
  > 📍 제주특별자치도 제주시 ○○로 ○○

**저녁**
- 장소 D(19시): 설명 내용
  > 📍 제주특별자치도 서귀포시 ○○로 ○○
- 장소 E(21시): 설명 내용
  > 📍 제주특별자치도 제주시 ○○로 ○○

2일차:
...

[실제 태스크]
아래 정보를 바탕으로, 위 형식대로 제주도 일정을 구성하세요.

**입력 정보:**
- 사용자 프로필: {user_profile.get_summary()}
- 최근 대화 내용: {history_summary or "첫 질문입니다"}
- 숙박 정보: {json.dumps([{"name": h.get("name", ""), "address": h.get("address", ""), "description": str(h.get("content") or h.get("description") or "")} for h in hotel_results[:hotel_count]], ensure_ascii=False)}
- 관광 정보: {json.dumps([{"name": t.get("name", ""), "address": t.get("address", ""), "description": str(t.get("content") or t.get("description") or "")} for t in travel_results[:tour_count]], ensure_ascii=False)}
- 음식 정보: {json.dumps([{"name": f.get("name", ""), "address": f.get("address", ""), "description": str(f.get("content") or f.get("description") or "")} for f in food_results[:food_count]], ensure_ascii=False)}

**작성 지침:**
- 사용자 성향과 대화 맥락을 반영해 **개인화된 일정**을 작성하세요.
- 시간대별로 **1~2개 장소**를 추천하며, **아침/점심/저녁 식사 장소는 반드시 포함**하세요.
- **관광 목적의 카페는 하루 1개까지만** 포함하세요.
- **장소 설명은 제공된 정보만 사용**하고, 추측은 절대 하지 마세요.
- **모든 장소는 정확한 이름과 주소를 반드시 포함**하여 작성하세요.
- **1일차 오후에 숙소 체크인**, 모든 날은 **숙소에서 마무리**, 마지막 날은 **공항에서 마무리**되도록 하세요.
"""
    
    try:
        # 복잡한 일정 생성을 위한 넉넉한 타임아웃 (120초)
        response = await asyncio.wait_for(
            response_llm.ainvoke(prompt), 
            timeout=120.0
        )
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

추출 가이드:
- "여자친구랑", "남친이랑", "연인과" → group_type: "커플"
- "2박3일", "3박4일" → duration: 그대로 추출
- "액티비티 좋아해", "맛집 찾아다니고" → interests 배열에 추가
- "서귀포", "제주시", "중문" → travel_region으로 추출
- "일정 짜달라", "추천해줘" → 별도 정보 없으면 null

명시적으로 언급된 정보만 추출해주세요."""

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
    # 6개 정보 중 3개 이상 있으면 일정 생성 가능 (데모용 개선)
    required_info_count = sum([
        bool(profile.travel_dates),      # 여행 날짜
        bool(profile.duration),          # 기간
        bool(profile.group_type),        # 여행 유형
        bool(profile.interests),         # 관심사
        bool(profile.budget),            # 예산
        bool(profile.travel_region)      # 여행 지역
    ])
    
    # 최소 3개 이상의 정보가 있으면 검색 시작
    result = required_info_count >= 3
    
    print(f"🧪 프로필 충분성 판단: {result} (필요정보: {required_info_count}/6)")
    return result

async def generate_info_collection_response(profile: UserProfile, user_message: str, conversation_history: List[Dict] = None) -> str:
    """정보 수집 응답 생성"""
    
    # 대화 히스토리 요약
    history_context = ""
    if conversation_history and len(conversation_history) > 1:
        recent_user_messages = [msg for msg in conversation_history[-4:] if msg['role'] == 'user']
        if recent_user_messages:
            history_context = f"\n이전 대화: {', '.join([msg['message'][:50] for msg in recent_user_messages])}"
    
    prompt = f"""제주도 여행 상담사로서 사용자와 자연스럽게 대화하면서 필요한 정보를 수집해주세요.

현재 수집된 정보: {profile.get_summary()}
사용자 최신 메시지: {user_message}{history_context}

**응답 가이드:**
- 이미 언급된 정보는 다시 묻지 않기
- 부족한 핵심 정보(여행 기간, 여행 유형, 관심사 등) 자연스럽게 확인
- 강요하지 않고 대화 맥락에 맞게 정보 수집
- 친근하고 도움되는 톤 유지
- 현재 정보로도 추천 가능함을 안내"""

    try:
        response = await profile_llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"❌ 정보 수집 응답 생성 오류: {e}")
        return "제주도 여행에 대해 더 자세히 알려주시면 더 좋은 추천을 드릴 수 있어요! 😊"

async def search_vector_db(query: str, category: str = "", top_k: int = 5) -> List[Dict]:
    """벡터 DB 검색 (재시도 및 백오프 로직 포함)"""
    max_retries = 3
    base_timeout = 90.0  # 대용량 요청을 위한 충분한 타임아웃
    
    for attempt in range(max_retries):
        try:
            # 재시도마다 타임아웃 증가 (90초 → 180초 → 270초)
            current_timeout = base_timeout * (attempt + 1)
            timeout_config = httpx.Timeout(
                connect=10.0, 
                read=current_timeout, 
                write=10.0, 
                pool=10.0
            )
            
            print(f"🔄 벡터 검색 시도 {attempt + 1}/{max_retries} - 타임아웃: {current_timeout}초")
            
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                # 여행 기간에 맞는 동적 검색 개수
                search_payload = {
                    "query": query,
                    "top_k": top_k,  # 여행 기간별 동적 개수
                    "search_type": "mmr",  # 다양성을 고려한 MMR 검색
                    "diversity_lambda": 0.5  # 유사성:다양성 = 50:50
                }
                
                response = await client.post(RAG_URL, json=search_payload)
                
                if response.status_code == 200:
                    result = response.json()
                    sources = result.get("sources", [])
                    processing_time = result.get("processing_time", 0)
                    
                    print(f"✅ 검색 성공 - {len(sources)}개 결과, {processing_time:.2f}초 소요 (요청: {top_k}개)")
                    
                    # sources와 answer 모두 확인 (간단 버전)
                    if sources and len(sources) > 0:
                        print(f"🧪 첫 번째 결과: {sources[0].get('content', '')[:100]}...")
                    
                    return sources[:top_k]  # 요청한 개수만큼 반환
                else:
                    print(f"❌ HTTP 오류 - 상태코드: {response.status_code}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # 지수 백오프
                        continue
                    return []
                
        except httpx.ReadTimeout:
            print(f"⏰ ReadTimeout 발생 ({current_timeout}초) - 시도 {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                print(f"🔄 {2 ** attempt}초 후 재시도...")
                await asyncio.sleep(2 ** attempt)  # 지수 백오프: 1초, 2초, 4초
                continue
            else:
                print("❌ 모든 재시도 실패 - 빈 결과 반환")
                return []
                
        except httpx.ConnectTimeout:
            print(f"🔌 연결 타임아웃 - RAG 서버 연결 실패")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return []
            
        except Exception as e:
            print(f"❌ 벡터 DB 검색 오류 - {type(e).__name__}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return []
    
    return []

# 여행 기간별 검색 개수 계산
def calculate_search_counts(duration: str) -> Dict[str, int]:
    """여행 기간에 따라 카테고리별 검색 개수 결정"""
    if not duration:
        return {"hotel": 3, "tour": 8, "food": 6, "event": 3}
    
    duration_lower = duration.lower()
    
    # 숫자 추출 (1박2일, 3박4일 등)
    import re
    numbers = re.findall(r'\d+', duration_lower)
    
    if numbers:
        # 가장 큰 숫자를 기준으로 (보통 총 일수)
        days = max(int(num) for num in numbers)
    else:
        # 텍스트 기반 판단
        if any(word in duration_lower for word in ['당일', '하루']):
            days = 1
        elif any(word in duration_lower for word in ['1박', '2일']):
            days = 2
        elif any(word in duration_lower for word in ['2박', '3일']):
            days = 3
        elif any(word in duration_lower for word in ['3박', '4일']):
            days = 4
        elif any(word in duration_lower for word in ['4박', '5일']):
            days = 5
        else:
            days = 3  # 기본값
    
    # 일수별 검색 개수 설정
    if days <= 1:
        counts = {"hotel": 3, "tour": 4, "food": 3, "event": 3}
    elif days <= 2:
        counts = {"hotel": 3, "tour": 6, "food": 5, "event": 3}
    elif days <= 3:
        counts = {"hotel": 4, "tour": 8, "food": 7, "event": 3}
    elif days <= 4:
        counts = {"hotel": 4, "tour": 12, "food": 10, "event": 3}
    elif days <= 5:
        counts = {"hotel": 5, "tour": 15, "food": 13, "event": 3}
    else:  # 6일 이상
        counts = {"hotel": 5, "tour": 18, "food": 16, "event": 3}
    
    print(f"📊 여행 기간 '{duration}' → {days}일 → 검색 개수: {counts}")
    return counts

# 큰 검색을 여러 번으로 분할하는 함수
async def search_with_batching(query: str, category: str, total_count: int, batch_size: int = 6) -> List[Dict]:
    """큰 검색 요청을 여러 번으로 나누어 처리 - 하지만 중복 문제로 인해 직접 처리 우선"""
    print(f"🧪 [BATCH_DEBUG] {category}: total_count={total_count}, batch_size={batch_size}")
    
    # 중복 문제를 피하기 위해 가능하면 직접 처리 (타임아웃 증가로 20개까지 가능)
    if total_count <= 20:  # 20개까지는 분할 안함
        print(f"🧪 [BATCH_DEBUG] {category}: 중복 방지 - 직접 처리 (≤20개)")
        return await search_vector_db(query, category, top_k=total_count)
    
    print(f"🔄 {category} 대량 검색: {total_count}개를 {batch_size}개씩 나누어 처리")
    print(f"🧪 [BATCH_DEBUG] {category}: 분할 처리 시작!")
    
    all_results = []
    batches_needed = (total_count + batch_size - 1) // batch_size  # 올림 계산
    
    for batch_num in range(batches_needed):
        try:
            current_batch_size = min(batch_size, total_count - len(all_results))
            print(f"📝 {category} 배치 {batch_num + 1}/{batches_needed}: {current_batch_size}개 요청")
            
            # 배치별로 약간 다른 쿼리로 다양성 확보
            if batch_num == 0:
                batch_query = query
            elif batch_num == 1:
                batch_query = query.replace("추천", "명소 리스트")
            else:
                batch_query = query.replace("추천", f"베스트 {batch_num + 1}")
            
            batch_results = await search_vector_db(batch_query, f"{category}_batch{batch_num+1}", top_k=current_batch_size)
            
            # 중복 제거 (이름 기준)
            existing_names = {result.get('name', '') for result in all_results}
            new_results = [result for result in batch_results if result.get('name', '') not in existing_names]
            
            all_results.extend(new_results)
            print(f"✅ {category} 배치 {batch_num + 1} 완료: {len(new_results)}개 추가 (중복 제거 후)")
            
            # 목표 달성 시 중단
            if len(all_results) >= total_count:
                break
            
            # 절반 이상 확보했고 타임아웃 위험이 있으면 조기 종료
            if len(all_results) >= total_count // 2 and batch_num >= 2:
                print(f"🎯 {category}: 충분한 결과 확보 ({len(all_results)}개) - 조기 완료")
                break
                
            # 서버 부하 방지를 위한 대기 (더 길게)
            if batch_num < batches_needed - 1:
                await asyncio.sleep(2.0)
                
        except Exception as e:
            print(f"❌ {category} 배치 {batch_num + 1} 실패: {e}")
            continue
    
    final_results = all_results[:total_count]  # 요청한 개수만큼만 반환
    print(f"🎯 {category} 최종 결과: {len(final_results)}개 (목표: {total_count}개)")
    return final_results

# 병렬 검색 기능 (여행 기간별 최적화)
async def parallel_search_all(state: GraphState) -> GraphState:
    """모든 카테고리를 병렬로 검색 (여행 기간별 개수 최적화)"""
    user_profile = state["user_profile"]
    
    # 여행 기간에 따른 검색 개수 결정
    search_counts = calculate_search_counts(user_profile.duration)
    
    # 각 카테고리별 LLM 기반 맞춤형 쿼리 생성 (세밀한 프롬프트 반영)
    print("🔍 각 카테고리별 맞춤형 쿼리 생성 중...")
    
    async def generate_hotel_query(profile):
        """숙박 검색 쿼리 생성 (개별 에이전트 프롬프트 사용)"""
        prompt = f"""당신은 제주 여행자를 위한 **숙박 검색 쿼리 생성 전문가**입니다.

사용자 프로필 정보를 참고해, 사용자의 관심사, 여행 지역, 여행 기간 정보를 바탕으로 **벡터 DB에서 숙박을 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 생성해주세요.

사용자 프로필: {profile.get_summary()}

쿼리에는 "제주도", "숙박", "호텔" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영해. (예: 감성 숙소, 자연 속 힐링, 오션뷰 숙소, 독채 숙소, 프라이빗 풀빌라 등)
- 관심사가 없는 경우 동행자 정보에 따라 장소의 분위기나 성격을 유추해서 적당한 표현을 넣어줘
    - **연인**이면 로맨틱하고 감성적인 숙소나 오션뷰 호텔
    - **가족**이면 아이 동반 가능한 가족형 리조트나 편의시설이 잘 갖춰진 곳
    - **친구**면 여러 명이 함께 묵을 수 있는 트렌디한 숙소나 감성 숙소
    - **혼자**면 조용하고 아늑한 1인 숙소나 자연과 가까운 힐링 공간

검색 쿼리:"""
        
        response = await hotel_llm.ainvoke(prompt)
        return response.content.strip()
    
    async def generate_event_query(profile):
        """이벤트 검색 쿼리 생성 (개별 에이전트 프롬프트 사용)"""
        prompt = f"""당신은 제주관광 전문 **자연어** **쿼리 생성 전문가**입니다.

다음과 같은 사용자 프로필을 참고하여, 벡터 DB에서 행사나 축제 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄을 만들어주세요.

사용자 프로필: {profile.get_summary()}

쿼리는 "제주도", "행사", "이벤트" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영해. (예: 로맨틱한 분위기, 트렌디한 분위기, 소규모 행사 등)
- 관심사가 없는 경우 동행자 정보나 지역을 바탕으로 자연스럽게 적절한 분위기나 스타일을 유추해줘.
    - **연인**이면 로맨틱하거나 분위기 좋은 곳
    - **가족**이면 다양한 연령대가 함께 즐기기 좋은 곳 
    - **친구**면 활기차고 활동적인 분위기의 축제나 트렌디한 행사
    - **혼자**면 조용히 즐길 수 있는 문화행사나 혼행객에게 인기 있는 소규모 지역 축제

자연어 검색 쿼리 한 문장으로 출력해주세요:"""
        
        response = await event_llm.ainvoke(prompt)
        return response.content.strip()
    
    async def generate_tour_query(profile):
        """관광지 검색 쿼리 생성 (개별 에이전트 프롬프트 사용)"""
        prompt = f"""당신은 제주관광 전문 **자연어** **쿼리 생성 전문가**입니다.

다음과 같은 사용자 프로필에서 사용자가 입력한 관심사, 여행 지역, 동행자 정보를 참고해서 **벡터 DB에서 관광지 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 만들어주세요.

사용자 프로필: {profile.get_summary()}

쿼리는 "제주도", "관광지" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영해. (예: 자연 풍경, 감성적인 장소, 사진 찍기 좋은 곳, 활동적인 체험, 전시 공간 등)
- 관심사가 없는 경우 동행자 정보에 따라 장소의 분위기나 성격을 유추해서 적당한 표현을 넣어줘
    - **연인**이면 감성적이거나 뷰가 좋은 데이트 코스
    - **가족**이면 아이와 함께 갈 수 있는 체험형 장소나 한적한 자연지
    - **친구**면 트렌디하고 재밌는 핫플
    - **혼자**면 조용히 걸을 수 있는 곳이나 분위기 있는 장소

검색 쿼리:"""
        
        response = await travel_llm.ainvoke(prompt)
        return response.content.strip()
    
    async def generate_food_query(profile):
        """음식점 검색 쿼리 생성 (개별 에이전트 프롬프트 사용)"""  
        prompt = f"""당신은 제주관광 전문 **자연어 쿼리 생성 전문가**입니다.

다음 사용자 프로필에서 사용자가 알려준 지역, 관심사, 그리고 동행자 정보를 참고해서 **벡터 DB에서 식당 또는 카페 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 만들어주세요

사용자 프로필: {profile.get_summary()}

쿼리는 "제주도", "맛집" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영합니다. (예: 감성적인 분위기, 현지인 맛집, 뷰 좋은 식당 등)
- 관심사가 없는 경우 동행자 정보나 지역을 바탕으로 자연스럽게 적절한 분위기나 음식 스타일을 유추합니다.
    - **연인**이면 로맨틱하거나 분위기 좋은 곳
    - **가족**이면 편하게 식사할 수 있는 한식이나 넓은 공간
    - **친구**면 캐주얼하거나 트렌디한 맛집
    - **혼자**면 조용하고 혼밥하기 좋은 곳

검색 쿼리:"""
        
        response = await food_llm.ainvoke(prompt)
        return response.content.strip()
    
    # 모든 카테고리에 LLM 기반 맞춤형 쿼리 생성
    hotel_query = await generate_hotel_query(user_profile)
    tour_query = await generate_tour_query(user_profile)
    food_query = await generate_food_query(user_profile)
    event_query = await generate_event_query(user_profile)
    
    queries = {
        "hotel": hotel_query,
        "tour": tour_query,
        "food": food_query,
        "event": event_query
    }
    
    print(f"🎯 생성된 맞춤형 쿼리들:")
    for category, query in queries.items():
        print(f"   {category}: '{query}'")
    
    print("🚀 순차 검색 시작 (동시 요청 문제 해결)...")
    
    # 모든 카테고리를 순차 처리 (RAG 서버 동시 요청 제한)
    categories = list(queries.items())
    results = {}
    
    # 1단계: hotel + tour 순차 검색 (동시 요청 문제 해결)
    print("📋 1단계: 숙박 + 관광지 순차 검색")
    for category, query in categories[:2]:  # hotel, tour
        count = search_counts.get(category, 5)
        print(f"📝 {category} 쿼리: '{query}' (검색 개수: {count}개)")
        
        # 순차 처리로 변경
        try:
            result = await search_with_batching(query, category, count, batch_size=3)
            results[category] = result
            print(f"🎯 {category} 순차 완료: {len(result)}개 결과 (목표: {count}개)")
        except Exception as e:
            print(f"❌ {category} 순차 검색 실패: {e}")
            results[category] = []
        
        # 각 검색 사이에 잠깐 대기
        await asyncio.sleep(1.0)
    
    # 더미 루프 (기존 코드 구조 유지)
    for category, task in []:
        try:
            expected_count = search_counts.get(category, 5)
            # 분할 처리 시간을 고려한 충분한 타임아웃 (배치수 * 최대시간)
            batches_needed = (expected_count + 2) // 3  # 3개씩 분할
            timeout = min(600.0, batches_needed * 150.0)  # 배치당 최대 150초  
            
            result = await asyncio.wait_for(task, timeout=timeout)
            results[category] = result
            print(f"🎯 {category} 최종 완료: {len(result)}개 결과 (목표: {expected_count}개)")
        except asyncio.TimeoutError:
            print(f"⏰ {category} 전체 타임아웃 ({timeout}초)")
            # 타임아웃이 발생해도 task에서 부분 결과를 얻을 수 있는지 확인
            try:
                if hasattr(task, 'result') and not task.cancelled():
                    partial_result = task.result()
                    results[category] = partial_result
                    print(f"🔄 {category} 부분 결과 확보: {len(partial_result)}개")
                else:
                    results[category] = []
                    print(f"❌ {category} 부분 결과 없음 - 빈 결과로 대체")
            except:
                results[category] = []
                print(f"❌ {category} 부분 결과 추출 실패 - 빈 결과로 대체")
        except Exception as e:
            print(f"❌ {category} 검색 실패: {e}")
            results[category] = []
    
    # 잠깐 대기 (서버 부하 감소) - 더 길게 대기
    await asyncio.sleep(5.0)
    
    # 2단계: food + event 순차 검색 (동시 요청 문제 해결)
    print("📋 2단계: 음식점 + 이벤트 순차 검색")
    for category, query in categories[2:]:  # food, event
        count = search_counts.get(category, 5)
        print(f"📝 {category} 쿼리: '{query}' (검색 개수: {count}개)")
        
        # 순차 처리로 변경
        try:
            result = await search_with_batching(query, category, count, batch_size=3)
            results[category] = result
            print(f"🎯 {category} 순차 완료: {len(result)}개 결과 (목표: {count}개)")
        except Exception as e:
            print(f"❌ {category} 순차 검색 실패: {e}")
            results[category] = []
        
        # 각 검색 사이에 잠깐 대기
        await asyncio.sleep(1.0)
    
    # 더미 루프 (기존 코드 구조 유지)
    for category, task in []:
        try:
            expected_count = search_counts.get(category, 5)
            # 분할 처리 시간을 고려한 충분한 타임아웃 (배치수 * 최대시간)
            batches_needed = (expected_count + 2) // 3  # 3개씩 분할
            timeout = min(600.0, batches_needed * 150.0)  # 배치당 최대 150초
            
            result = await asyncio.wait_for(task, timeout=timeout)
            results[category] = result
            print(f"🎯 {category} 최종 완료: {len(result)}개 결과 (목표: {expected_count}개)")
        except asyncio.TimeoutError:
            print(f"⏰ {category} 전체 타임아웃 ({timeout}초)")
            # 타임아웃이 발생해도 task에서 부분 결과를 얻을 수 있는지 확인
            try:
                if hasattr(task, 'result') and not task.cancelled():
                    partial_result = task.result()
                    results[category] = partial_result
                    print(f"🔄 {category} 부분 결과 확보: {len(partial_result)}개")
                else:
                    results[category] = []
                    print(f"❌ {category} 부분 결과 없음 - 빈 결과로 대체")
            except:
                results[category] = []
                print(f"❌ {category} 부분 결과 추출 실패 - 빈 결과로 대체")
        except Exception as e:
            print(f"❌ {category} 검색 실패: {e}")
            results[category] = []
    
    return {
        **state,
        "hotel_results": results.get("hotel", []),
        "travel_results": results.get("tour", []),
        "food_results": results.get("food", []),
        "event_results": results.get("event", [])
    }

# 조건부 라우팅 함수
def should_continue_to_agents(state: GraphState) -> str:
    """프로필이 준비되었는지 확인하여 다음 단계 결정"""
    if state.get("profile_ready", False):
        return "parallel_search"  # 병렬 검색으로 변경
    else:
        return "end"

def should_continue_to_response(state: GraphState) -> str:
    """병렬 검색 후 응답 생성으로 이동"""
    return "response_generator"

# 성능 진단 함수
async def diagnose_rag_server() -> Dict:
    """RAG 서버 상태 및 성능 진단"""
    try:
        print("🔍 RAG 서버 진단 시작...")
        
        timeout_config = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            start_time = asyncio.get_event_loop().time()
            
            # 간단한 테스트 쿼리 (RAG 서버는 /chat 엔드포인트만 지원)
            response = await client.post(
                RAG_URL,  # 직접 /chat 엔드포인트 사용
                json={"query": "제주도"}
            )
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            result = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": f"{response_time:.2f}초",
                "server_url": RAG_URL
            }
            
            if response.status_code == 200:
                print(f"✅ RAG 서버 정상 - 응답시간: {response_time:.2f}초")
            else:
                print(f"⚠️ RAG 서버 응답 이상 - 상태코드: {response.status_code}")
            
            return result
            
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "server_url": RAG_URL
        }
        print(f"❌ RAG 서버 진단 실패: {e}")
        return error_result

# LangGraph 설정
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("profile_collector", profile_collector_node)
workflow.add_node("parallel_search", parallel_search_all)  # 병렬 검색 노드 추가
workflow.add_node("response_generator", response_generator_node)

# 기존 개별 에이전트들은 유지 (필요시 사용)
workflow.add_node("hotel_agent", hotel_agent_node)
workflow.add_node("travel_agent", travel_agent_node)
workflow.add_node("food_agent", food_agent_node)
workflow.add_node("event_agent", event_agent_node)

# 시작점 설정
workflow.set_entry_point("profile_collector")

# 조건부 엣지 설정 (병렬 검색으로 라우팅)
workflow.add_conditional_edges(
    "profile_collector",
    should_continue_to_agents,
    {
        "parallel_search": "parallel_search",  # 병렬 검색으로 변경
        "end": END
    }
)

# 병렬 검색 → 응답 생성
workflow.add_conditional_edges(
    "parallel_search",
    should_continue_to_response,
    {
        "response_generator": "response_generator"
    }
)

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
        config = {"configurable": {"thread_id": self.session_id}}
        
        try:
            # 기존 상태 불러오기 (메모리에서)
            try:
                current_state = await self.graph.aget_state(config)
                if current_state and current_state.values:
                    # 기존 상태가 있으면 사용자 메시지만 업데이트
                    state = current_state.values.copy()
                    state["user_message"] = user_message
                    print(f"🔄 기존 상태 불러옴 - 대화 기록: {len(state.get('conversation_history', []))}개")
                else:
                    # 기존 상태가 없으면 새로 생성
                    state = {
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
                    print(f"🆕 새로운 상태 생성")
            except Exception as e:
                print(f"⚠️ 상태 불러오기 실패, 새로 생성: {e}")
                state = {
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
            
            # 그래프 실행
            result = await self.graph.ainvoke(state, config)
            
            # 응답과 프로필 정보 반환
            response_text = result.get("final_response", "죄송합니다. 응답을 생성할 수 없습니다.")
            user_profile = result.get("user_profile", UserProfile())
            
            return {
                "response": response_text,
                "user_profile": user_profile
            }
            
        except Exception as e:
            print(f"❌ 챗봇 실행 오류: {e}")
            return {
                "response": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
                "user_profile": UserProfile()
            }

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
    content: str  # backend에서 'content' 필드로 전송
    session_id: str
    conversation_history: List[Dict] = []
    user_profile: Dict = {}
    profile_completion: float = 0.0

class ChatResponse(BaseModel):
    response: str
    session_id: str
    needs_more_info: bool = False
    profile_completion: float = 0.0
    follow_up_questions: List[str] = []
    user_profile: Dict = {}
    analysis_confidence: float = 0.8
    timestamp: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 엔드포인트"""
    
    try:
        if request.session_id:
            chatbot.session_id = request.session_id
            
        result = await chatbot.chat(request.content)
        
        # 프로필 정보를 딕셔너리로 변환
        profile_dict = {}
        if result.get("user_profile"):
            profile = result["user_profile"]
            profile_dict = {
                "travel_dates": profile.travel_dates,
                "duration": profile.duration,
                "group_type": profile.group_type,
                "interests": profile.interests,
                "budget": profile.budget,
                "travel_region": profile.travel_region
            }
        
        # 프로필 완성도 계산
        profile_completion = 0.0
        if profile_dict:
            completed_fields = sum(1 for v in profile_dict.values() if v)
            profile_completion = completed_fields / len(profile_dict)
        
        # 더 많은 정보가 필요한지 판단
        needs_more_info = profile_completion < 0.8
        
        return ChatResponse(
            response=result["response"],
            session_id=request.session_id or "default",
            needs_more_info=needs_more_info,
            profile_completion=profile_completion,
            follow_up_questions=[],
            user_profile=profile_dict,
            analysis_confidence=0.8,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"❌ 채팅 오류: {e}")
        return ChatResponse(
            response="죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
            session_id=request.session_id or "default",
            needs_more_info=True,
            profile_completion=0.0,
            follow_up_questions=[],
            user_profile={},
            analysis_confidence=0.0,
            timestamp=datetime.now().isoformat()
        )

@app.get("/")
async def root():
    return {"message": "🌴 LangGraph 기반 제주도 멀티 에이전트 챗봇 API"}

@app.get("/health")
async def health_check():
    """시스템 상태 및 RAG 서버 진단"""
    rag_diagnosis = await diagnose_rag_server()
    
    return {
        "chatbot_status": "healthy",
        "rag_server": rag_diagnosis,
        "timestamp": datetime.now().isoformat(),
        "features": {
            "parallel_search": True,
            "retry_logic": True,
            "timeout_handling": True,
            "memory_support": True
        }
    }

@app.get("/performance-tips")
async def performance_tips():
    """ReadTimeout 문제 해결 팁"""
    return {
        "readtimeout_solutions": {
            "1_retry_logic": "자동 재시도 (3회) + 지수 백오프",
            "2_parallel_search": "병렬 검색으로 전체 시간 단축",  
            "3_timeout_escalation": "30초 → 60초 → 90초 점진적 증가",
            "4_performance_optimization": "similarity 검색 (MMR 대신)",
            "5_dynamic_count": "여행 기간별 동적 검색 개수 조정"
        },
        "smart_search_counts": {
            "1_day": {"hotel": 3, "tour": 4, "food": 3, "event": 2},
            "2_days": {"hotel": 3, "tour": 6, "food": 5, "event": 3},
            "3_days": {"hotel": 4, "tour": 8, "food": 6, "event": 3},
            "4_days": {"hotel": 4, "tour": 12, "food": 8, "event": 4},
            "5_days": {"hotel": 5, "tour": 15, "food": 10, "event": 5},
            "6+_days": {"hotel": 5, "tour": 18, "food": 12, "event": 6}
        },
        "benefits": {
            "comprehensive_itinerary": "여행 기간에 맞는 충분한 장소 정보",
            "daily_distribution": "각 날짜별 적절한 관광지/음식점 배분",
            "no_repetition": "호텔 근처에서만 먹지 않고 다양한 지역 탐방",
            "realistic_scheduling": "하루 2-3곳 관광지로 현실적인 일정"
        },
        "server_optimization": {
            "rag_server": "advanced_jeju_chatbot/api/main.py",
            "vector_db": "ChromaDB 인덱스 최적화 필요시 재구축",
            "llm_api": "Upstage Solar Pro API 응답 시간 모니터링",
            "adaptive_timeout": "검색 개수에 따른 동적 타임아웃 조정"
        },
        "monitoring": {
            "health_check": "/health 엔드포인트 사용",
            "processing_time": "각 검색별 소요 시간 표시",
            "error_logging": "상세한 오류 타입 및 메시지 제공",
            "search_count_display": "카테고리별 요청/실제 결과 수 표시"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 LangGraph 제주도 멀티 에이전트 챗봇 시작!")
    print("📍 서버: http://localhost:8001")
    print("🔍 진단: http://localhost:8001/health")
    print("💡 성능 팁: http://localhost:8001/performance-tips")
    uvicorn.run(app, host="0.0.0.0", port=8001)  # backend에서 8001 포트로 호출 