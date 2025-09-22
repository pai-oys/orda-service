"""
🌴 데모데이용 개인화된 제주도 여행 챗봇
- 사용자 이름 기반 성향과 여행 스타일 자동 적용
- 성향별 맞춤 말투로 응답
- 여행 스타일에 맞는 일정 추천
"""

import asyncio
import httpx
import json
from typing import Dict, List, Optional, TypedDict, Annotated
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

# 사용자 데이터 로딩
def load_user_data():
    """데모용 사용자 데이터 로딩"""
    try:
        with open('demo_user_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {user['name']: user for user in data['users']}
    except FileNotFoundError:
        print("demo_user_data.json 파일을 찾을 수 없습니다.")
        return {}

USER_DATA = load_user_data()

@dataclass
class PersonalizedUserProfile:
    """개인화된 사용자 프로필 정보"""
    name: Optional[str] = None
    personality: Optional[str] = None  # 에겐남, 에겐녀, 테토남, 테토녀
    travel_style: Optional[str] = None
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
        if self.name:
            summary_parts.append(f"이름: {self.name}")
        if self.personality:
            summary_parts.append(f"성향: {self.personality}")
        if self.travel_style:
            summary_parts.append(f"여행 스타일: {self.travel_style}")
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
    
    def is_sufficient(self) -> bool:
        """프로필이 여행 계획을 위해 충분한지 판단 - 간단하게 3개 정보만 있으면 OK"""
        # 모든 가능한 정보들
        all_fields = [
            self.duration,       # 여행 기간
            self.group_type,     # 동행자
            self.travel_region,  # 여행 지역
            bool(self.interests), # 관심사가 있는지
            self.name,           # 이름
            self.personality,    # 성향
            self.travel_style    # 여행 스타일
        ]
        
        # 정보가 3개 이상 있으면 충분
        filled_count = sum(1 for field in all_fields if field)
        return filled_count >= 3

# LangGraph State 정의  
class PersonalizedGraphState(TypedDict):
    """개인화된 그래프 상태"""
    user_message: Annotated[str, lambda x, y: y or x]  # 새 값이 있으면 새 값 사용, 없으면 기존 값 유지
    conversation_history: List[Dict]
    user_profile: PersonalizedUserProfile
    hotel_results: List[Dict]
    travel_results: List[Dict] 
    food_results: List[Dict]
    event_results: List[Dict]
    final_response: str
    profile_ready: bool

# 성향별 말투 정의
PERSONALITY_STYLES = {
    "에겐남": {
        "tone": "따뜻하고 배려심 넘치는 말투",
        "characteristics": "상대방을 진심으로 걱정하고 배려하는 표현, 부드럽고 친근한 말투",
        "example_phrases": ["정말 좋은 선택이실 것 같아요", "혹시 괜찮으시다면", "편하게 말씀해주세요", "마음에 드셨으면 좋겠어요", "걱정 마세요"]
    },
    "에겐녀": {
        "tone": "다정하고 상냥한 말투",
        "characteristics": "따뜻하고 섬세한 표현, 상대방의 감정을 세심하게 배려하는 말투",
        "example_phrases": ["정말 예쁠 것 같아요♡", "마음이 편안해지실 거예요", "너무 로맨틱할 것 같아요", "기분 좋아지실 거예요", "힐링되실 거예요"]
    },
    "테토남": {
        "tone": "직설적이고 거침없는 말투",
        "characteristics": "솔직하고 단도직입적인 표현, 약간 무뚝뚝하지만 확신에 찬 말투",
        "example_phrases": ["이거 진짜 좋음", "그냥 여기 가", "확실함", "이거 아니면 말고", "뭘 고민해", "당연히 이거지"]
    },
    "테토녀": {
        "tone": "솔직하고 시원시원한 말투",
        "characteristics": "직설적이고 당당한 표현, 약간 쿨하고 드라이한 말투",
        "example_phrases": ["이거 레전드임", "진짜 개좋아", "확실히 갈 만함", "이거 아니면 뭐함", "당연히 여기지", "완전 인정"]
    }
}

# 공용 LLM 인스턴스들
profile_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
hotel_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
travel_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
food_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
event_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
response_llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")

# 벡터 DB 접근 URL
RAG_URL = "http://localhost:8002/chat"

def get_user_info_by_name(name: str) -> Dict:
    """이름으로 사용자 정보 조회"""
    return USER_DATA.get(name, {})

def create_personality_prompt(personality: str, travel_style: str) -> str:
    """성향별 맞춤 프롬프트 생성"""
    if personality not in PERSONALITY_STYLES:
        return ""
    
    style_info = PERSONALITY_STYLES[personality]
    
    return f"""
당신은 {personality} 성향의 제주도 여행 전문 상담사입니다.

성향 특징:
- {style_info['characteristics']}
- {style_info['tone']}
- 예시 표현: {', '.join(style_info['example_phrases'])}

사용자의 여행 스타일: {travel_style}

이 성향과 여행 스타일에 맞게 자연스럽게 대화하고, 여행 일정을 추천할 때는 반드시 사용자의 여행 스타일을 고려해주세요.
"""

# 개인화된 프로필 수집 노드 (기존 로직 + 개인화)
async def personalized_profile_collector_node(state: PersonalizedGraphState) -> PersonalizedGraphState:
    """개인화된 사용자 프로필 수집 및 업데이트 (기존 smart_chatbot.py 로직 사용)"""
    user_message = state["user_message"]
    conversation_history = state.get("conversation_history", [])
    current_profile = state.get("user_profile", PersonalizedUserProfile())
    
    # 대화 기록에 사용자 메시지 추가
    conversation_history.append({
        "role": "user", 
        "message": user_message,
        "timestamp": datetime.now().isoformat()
    })
    
    # 이름 추출 시도 (개인화 부분)
    if not current_profile.name:
        for name in USER_DATA.keys():
            if name in user_message:
                user_info = get_user_info_by_name(name)
                current_profile.name = name
                current_profile.personality = user_info.get('personality')
                current_profile.travel_style = user_info.get('travel_style')
                print(f"🎯 개인화 정보 설정: {name} ({current_profile.personality}) - {current_profile.travel_style}")
                break
    
    # 프로필 정보 추출 (기존 로직)
    profile_info = await extract_personalized_profile_info(user_message, current_profile)
    print(f"🔍 추출된 프로필 정보: {profile_info}")
    
    # 프로필 업데이트 (기존 로직)
    updated_profile = update_personalized_profile(current_profile, profile_info)
    print(f"📝 업데이트된 프로필: {updated_profile.get_summary()}")
    
    # 프로필이 충분한지 확인 (기존 로직)
    profile_ready = is_personalized_profile_sufficient(updated_profile)
    
    if not profile_ready:
        # 추가 정보 수집 응답 생성 (개인화 적용)
        response = await generate_personalized_info_collection_response(updated_profile, user_message, conversation_history)
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

# 개인화된 프로필 정보 추출 함수
async def extract_personalized_profile_info(message: str, current_profile: PersonalizedUserProfile) -> Dict:
    """메시지에서 프로필 정보 추출 (기존 로직 + 개인화)"""
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
        print(f"❌ 개인화 프로필 추출 오류: {e}")
        return {}

def update_personalized_profile(current_profile: PersonalizedUserProfile, profile_info: Dict) -> PersonalizedUserProfile:
    """개인화된 프로필 업데이트"""
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

def is_personalized_profile_sufficient(profile: PersonalizedUserProfile) -> bool:
    """개인화된 프로필이 충분한지 확인"""
    # 개인화된 경우 이름+성향이 있으면 기본적으로 충분
    if profile.name and profile.personality:
        return True
    
    # 기존 로직: 여행 날짜나 기간이 있으면 충분
    return bool(profile.travel_dates or profile.duration)

async def generate_personalized_info_collection_response(profile: PersonalizedUserProfile, user_message: str, conversation_history: List[Dict]) -> str:
    """개인화된 추가 정보 수집 응답 생성"""
    personality_context = ""
    if profile.personality and profile.travel_style:
        style_info = PERSONALITY_STYLES.get(profile.personality, {})
        personality_context = f"""
당신은 {profile.personality} 성향입니다:
- {style_info.get('tone', '자연스러운 말투')}로 대화하세요.
- {style_info.get('characteristics', '친근하고 도움이 되는 표현')}을 사용하세요.
"""
    
    prompt = f"""당신은 제주도 여행 상담사입니다.

{personality_context}

현재 사용자 프로필: {profile.get_summary()}
사용자 메시지: {user_message}

아직 부족한 정보를 자연스럽게 물어보는 응답을 생성해주세요.
필요한 정보: 여행 날짜, 기간, 동행자, 관심사 등

친근하고 자연스럽게 대화하면서 정보를 수집하세요."""

    try:
        response = await profile_llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"❌ 개인화 정보 수집 응답 생성 오류: {e}")
        return "여행 계획을 위해 몇 가지 정보가 더 필요해요. 언제, 며칠 정도 여행하실 예정인가요?"

# 벡터 DB 검색 함수 (smart_chatbot.py와 동일)
async def search_vector_db(query: str, category: str = "", top_k: int = 5) -> List[Dict]:
    """벡터 DB 검색 (재시도 및 백오프 로직 포함) - smart_chatbot.py와 동일"""
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
                    else:
                        print(f"❌ 최대 재시도 횟수 초과 - 빈 결과 반환")
                        return []
                        
        except Exception as e:
            print(f"❌ 벡터 검색 오류 (시도 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 지수 백오프
                continue
            else:
                print(f"❌ 최대 재시도 횟수 초과 - 빈 결과 반환")
                return []
    
    return []

# 숙박 에이전트 노드 (기존 로직 + 개인화)
async def hotel_agent_node(state: PersonalizedGraphState) -> PersonalizedGraphState:
    """숙박 장소 검색 에이전트 (개인화 적용)"""
    user_profile = state["user_profile"]
    
    # 개인화 정보를 반영한 검색 쿼리 생성 프롬프트
    personality_context = ""
    if user_profile.personality and user_profile.travel_style:
        personality_context = f"""
사용자 성향: {user_profile.personality}
여행 스타일: {user_profile.travel_style}
위 정보를 고려하여 성향에 맞는 숙소를 찾을 수 있도록 검색 쿼리를 생성해주세요.
"""
    
    prompt = f"""당신은 제주 여행자를 위한 **숙박 검색 쿼리 생성 전문가**입니다.

사용자 프로필 정보를 참고해, 사용자의 관심사, 여행 지역, 여행 기간 정보를 바탕으로 **벡터 DB에서 숙박을 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 생성해주세요.

사용자 프로필: {user_profile.get_summary()}

{personality_context}

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
        print(f"🏨 개인화 숙박 에이전트 쿼리: '{search_query}'")
        
        # 벡터 DB 검색 (smart_chatbot.py와 동일)
        hotel_results = await search_vector_db(search_query, "hotel", top_k=5)
        
        print(f"🏨 숙박 검색 결과 ({len(hotel_results)}개)")
        
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

# 관광지 에이전트 노드 (기존 로직 + 개인화)
async def travel_agent_node(state: PersonalizedGraphState) -> PersonalizedGraphState:
    """관광지 검색 에이전트 (개인화 적용)"""
    user_profile = state["user_profile"]
    
    personality_context = ""
    if user_profile.personality and user_profile.travel_style:
        personality_context = f"""
사용자 성향: {user_profile.personality}
여행 스타일: {user_profile.travel_style}
위 정보를 고려하여 성향에 맞는 관광지를 찾을 수 있도록 검색 쿼리를 생성해주세요.
"""
    
    prompt = f"""당신은 제주 여행자를 위한 **관광지 검색 쿼리 생성 전문가**입니다.

사용자 프로필 정보를 참고해, 사용자의 관심사, 여행 지역, 여행 기간 정보를 바탕으로 **벡터 DB에서 관광지를 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 생성해주세요.

사용자 프로필: {user_profile.get_summary()}

{personality_context}

쿼리에는 "제주도", "관광", "여행지" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영해. (예: 자연 힐링, 감성 카페, 액티비티, 문화 체험, 바다 뷰 등)
- 관심사가 없는 경우 동행자 정보에 따라 장소의 분위기나 성격을 유추해서 적당한 표현을 넣어줘

검색 쿼리:"""
    
    try:
        response = await travel_llm.ainvoke(prompt)
        search_query = response.content.strip()
        print(f"🗺️ 개인화 관광지 에이전트 쿼리: '{search_query}'")
        
        travel_results = await search_vector_db(search_query, "travel", top_k=8)
        print(f"🗺️ 관광지 검색 결과 ({len(travel_results)}개)")
        
        return {**state, "travel_results": travel_results}
        
    except Exception as e:
        print(f"❌ 관광지 에이전트 오류: {e}")
        return {**state, "travel_results": []}

# 음식 에이전트 노드 (기존 로직 + 개인화)
async def food_agent_node(state: PersonalizedGraphState) -> PersonalizedGraphState:
    """음식 검색 에이전트 (개인화 적용)"""
    user_profile = state["user_profile"]
    
    personality_context = ""
    if user_profile.personality and user_profile.travel_style:
        personality_context = f"""
사용자 성향: {user_profile.personality}
여행 스타일: {user_profile.travel_style}
위 정보를 고려하여 성향에 맞는 맛집을 찾을 수 있도록 검색 쿼리를 생성해주세요.
"""
    
    prompt = f"""당신은 제주 여행자를 위한 **맛집 검색 쿼리 생성 전문가**입니다.

사용자 프로필 정보를 참고해, 사용자의 관심사, 여행 지역, 여행 기간 정보를 바탕으로 **벡터 DB에서 맛집을 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 생성해주세요.

사용자 프로필: {user_profile.get_summary()}

{personality_context}

쿼리에는 "제주도", "맛집", "음식" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영해. (예: 감성 카페, 로컬 맛집, 해산물, 흑돼지, 디저트 등)
- 관심사가 없는 경우 동행자 정보에 따라 장소의 분위기나 성격을 유추해서 적당한 표현을 넣어줘

검색 쿼리:"""
    
    try:
        response = await food_llm.ainvoke(prompt)
        search_query = response.content.strip()
        print(f"🍽️ 개인화 음식 에이전트 쿼리: '{search_query}'")
        
        food_results = await search_vector_db(search_query, "food", top_k=6)
        print(f"🍽️ 음식 검색 결과 ({len(food_results)}개)")
        
        return {**state, "food_results": food_results}
        
    except Exception as e:
        print(f"❌ 음식 에이전트 오류: {e}")
        return {**state, "food_results": []}

# 이벤트 에이전트 노드 (기존 로직 + 개인화)
async def event_agent_node(state: PersonalizedGraphState) -> PersonalizedGraphState:
    """이벤트 검색 에이전트 (개인화 적용)"""
    user_profile = state["user_profile"]
    
    personality_context = ""
    if user_profile.personality and user_profile.travel_style:
        personality_context = f"""
사용자 성향: {user_profile.personality}
여행 스타일: {user_profile.travel_style}
위 정보를 고려하여 성향에 맞는 이벤트를 찾을 수 있도록 검색 쿼리를 생성해주세요.
"""
    
    prompt = f"""당신은 제주 여행자를 위한 **이벤트 검색 쿼리 생성 전문가**입니다.

사용자 프로필 정보를 참고해, 사용자의 관심사, 여행 지역, 여행 기간 정보를 바탕으로 **벡터 DB에서 이벤트를 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 생성해주세요.

사용자 프로필: {user_profile.get_summary()}

{personality_context}

쿼리에는 "제주도", "이벤트", "축제", "체험" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

- 관심사가 있는 경우 그걸 자연스럽게 반영해. (예: 문화 축제, 체험 프로그램, 공연, 전시 등)
- 관심사가 없는 경우 동행자 정보에 따라 장소의 분위기나 성격을 유추해서 적당한 표현을 넣어줘

검색 쿼리:"""
    
    try:
        response = await event_llm.ainvoke(prompt)
        search_query = response.content.strip()
        print(f"🎪 개인화 이벤트 에이전트 쿼리: '{search_query}'")
        
        event_results = await search_vector_db(search_query, "event", top_k=3)
        print(f"🎪 이벤트 검색 결과 ({len(event_results)}개)")
        
        return {**state, "event_results": event_results}
        
    except Exception as e:
        print(f"❌ 이벤트 에이전트 오류: {e}")
        return {**state, "event_results": []}

# 개인화된 응답 생성 노드 (기존 로직 + 개인화 말투)
# 개인화된 병렬 검색 노드 (smart_chatbot.py 기반)
async def personalized_parallel_search_all(state: PersonalizedGraphState) -> PersonalizedGraphState:
    """모든 카테고리를 병렬로 검색 (개인화 버전)"""
    user_profile = state["user_profile"]
    
    # 여행 기간에 따른 검색 개수 결정 (smart_chatbot.py와 동일)
    search_counts = {
        "hotel": 3,
        "tour": 8, 
        "food": 6,
        "event": 3
    }
    
    print("🔍 개인화된 맞춤형 쿼리 생성 중...")
    
    # 개인화 컨텍스트 생성
    personality_context = ""
    if user_profile.personality and user_profile.travel_style:
        personality_info = PERSONALITY_STYLES.get(user_profile.personality, {})
        personality_context = f"\n사용자 성향: {user_profile.personality} - {personality_info.get('description', '')}\n여행 스타일: {user_profile.travel_style}\n"
    
    # 각 카테고리별 개인화된 쿼리 생성
    async def generate_personalized_hotel_query(profile):
        base_prompt = f"""당신은 제주 여행자를 위한 **숙박 검색 쿼리 생성 전문가**입니다.

사용자 프로필 정보를 참고해, 사용자의 관심사, 여행 지역, 여행 기간 정보를 바탕으로 **벡터 DB에서 숙박을 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 생성해주세요.

사용자 프로필: {profile.get_summary()}{personality_context}

쿼리에는 "제주도", "숙박", "호텔" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

검색 쿼리:"""
        
        response = await hotel_llm.ainvoke(base_prompt)
        return response.content.strip()
    
    async def generate_personalized_tour_query(profile):
        base_prompt = f"""당신은 제주관광 전문 **자연어** **쿼리 생성 전문가**입니다.

다음과 같은 사용자 프로필에서 사용자가 입력한 관심사, 여행 지역, 동행자 정보를 참고해서 **벡터 DB에서 관광지 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 만들어주세요.

사용자 프로필: {profile.get_summary()}{personality_context}

쿼리는 "제주도", "관광지" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

검색 쿼리:"""
        
        response = await travel_llm.ainvoke(base_prompt)
        return response.content.strip()
    
    async def generate_personalized_food_query(profile):
        base_prompt = f"""당신은 제주관광 전문 **자연어 쿼리 생성 전문가**입니다.

다음 사용자 프로필에서 사용자가 알려준 지역, 관심사, 그리고 동행자 정보를 참고해서 **벡터 DB에서 식당 또는 카페 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄**을 만들어주세요

사용자 프로필: {profile.get_summary()}{personality_context}

쿼리는 "제주도", "맛집" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

검색 쿼리:"""
        
        response = await food_llm.ainvoke(base_prompt)
        return response.content.strip()
    
    async def generate_personalized_event_query(profile):
        base_prompt = f"""당신은 제주관광 전문 **자연어** **쿼리 생성 전문가**입니다.

다음과 같은 사용자 프로필을 참고하여, 벡터 DB에서 행사나 축제 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄을 만들어주세요.

사용자 프로필: {profile.get_summary()}{personality_context}

쿼리는 "제주도", "행사", "이벤트" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.

자연어 검색 쿼리 한 문장으로 출력해주세요:"""
        
        response = await event_llm.ainvoke(base_prompt)
        return response.content.strip()
    
    # 모든 카테고리에 개인화된 쿼리 생성
    hotel_query = await generate_personalized_hotel_query(user_profile)
    tour_query = await generate_personalized_tour_query(user_profile)
    food_query = await generate_personalized_food_query(user_profile)
    event_query = await generate_personalized_event_query(user_profile)
    
    print(f"🎯 개인화된 맞춤형 쿼리들:")
    print(f"   🏨 숙박 쿼리: '{hotel_query}'")
    print(f"   🗺️ 관광지 쿼리: '{tour_query}'")
    print(f"   🍽️ 음식 쿼리: '{food_query}'")
    print(f"   🎉 이벤트 쿼리: '{event_query}'")
    
    # 순차 검색 (smart_chatbot.py와 동일)
    print("🚀 순차 검색 시작...")
    
    results = {}
    queries = [
        ("hotel", hotel_query, search_counts["hotel"]),
        ("tour", tour_query, search_counts["tour"]),
        ("food", food_query, search_counts["food"]),
        ("event", event_query, search_counts["event"])
    ]
    
    for category, query, count in queries:
        print(f"📝 {category} 쿼리: '{query}' (검색 개수: {count}개)")
        
        try:
            result = await search_vector_db(query, category, count)
            results[category] = result
            print(f"🎯 {category} 완료: {len(result)}개 결과")
        except Exception as e:
            print(f"❌ {category} 검색 실패: {e}")
            results[category] = []
        
        # 각 검색 사이에 잠깐 대기
        await asyncio.sleep(1.0)
    
    # 상태 업데이트
    state["hotel_results"] = results.get("hotel", [])
    state["travel_results"] = results.get("tour", [])
    state["food_results"] = results.get("food", [])
    state["event_results"] = results.get("event", [])
    
    print(f"📊 개인화 검색 완료 - 호텔: {len(state['hotel_results'])}개, 관광: {len(state['travel_results'])}개, 음식: {len(state['food_results'])}개, 이벤트: {len(state['event_results'])}개")
    
    return state

async def personalized_response_node(state: PersonalizedGraphState) -> PersonalizedGraphState:
    """개인화된 최종 응답 생성 (기존 smart_chatbot.py 로직 + 개인화)"""
    user_message = state["user_message"]
    user_profile = state["user_profile"]
    hotel_results = state.get("hotel_results", [])
    travel_results = state.get("travel_results", [])
    food_results = state.get("food_results", [])
    event_results = state.get("event_results", [])
    conversation_history = state.get("conversation_history", [])
    
    # 대화 히스토리 요약
    history_summary = None
    if len(conversation_history) > 1:
        recent_messages = conversation_history[-3:]
        history_summary = " | ".join([f"{msg['role']}: {msg['message'][:50]}..." for msg in recent_messages])
    
    # 여행 기간별 결과 활용량 결정 (기존 로직)
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
    
    print(f"📊 개인화 응답 생성용 정보 활용: 호텔 {hotel_count}개, 관광 {tour_count}개, 음식 {food_count}개, 이벤트 {event_count}개")
    
    # 개인화된 시스템 메시지 생성 (강력한 말투 반영)
    personality_instruction = ""
    if user_profile.personality and user_profile.travel_style:
        style_info = PERSONALITY_STYLES.get(user_profile.personality, {})
        example_phrases = style_info.get('example_phrases', [])
        
        personality_instruction = f"""
🎭 **{user_profile.personality} 성향 필수 적용!**

**말투**: {style_info.get('tone', '자연스러운 말투')}
**핵심 표현**: {', '.join(example_phrases[:2])}

{"**에겐**: 따뜻하고 자세한 설명, 감정 표현 풍부" if user_profile.personality in ["에겐남", "에겐녀"] else "**테토**: 간결하고 직설적, 핵심만 간단히"}
"""
    else:
        personality_instruction = "- 마치 친구처럼 친근하고 자연스러운 말투로 사용자에게 말하세요."
    
    prompt = f"""
[시스템 메시지]
당신은 제주 여행 일정 추천 전문가입니다.

{personality_instruction}
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
- 장소 A(11시): …

**오후**
- 장소 B(13시): …
- 장소 C(16시): …

**저녁**
- 장소 D(19시): …
- 장소 E(21시): …

2일차:
...

[실제 태스크]
아래 정보를 바탕으로, 위 형식대로 제주도 일정을 구성하세요.

**입력 정보:**
- 사용자 프로필: {user_profile.get_summary()}
- 최근 대화 내용: {history_summary or "첫 질문입니다"}
- 숙박 정보: {json.dumps([{"name": h.get("name", ""), "description": str(h.get("content") or h.get("description") or "")} for h in hotel_results[:hotel_count]], ensure_ascii=False)}
- 관광 정보: {json.dumps([{"name": t.get("name", ""), "description": str(t.get("content") or t.get("description") or "")} for t in travel_results[:tour_count]], ensure_ascii=False)}
- 음식 정보: {json.dumps([{"name": f.get("name", ""), "description": str(f.get("content") or f.get("description") or "")} for f in food_results[:food_count]], ensure_ascii=False)}

**작성 지침:**
- 사용자 성향과 대화 맥락을 반영해 **개인화된 일정**을 작성하세요.
- 시간대별로 **1~2개 장소**를 추천하며, **아침/점심/저녁 식사 장소는 반드시 포함**하세요.
- **관광 목적의 카페는 하루 1개까지만** 포함하세요.
- **1일차 오후에 숙소 체크인**, 모든 날은 **숙소에서 마무리**, 마지막 날은 **공항에서 마무리**되도록 하세요.

**장소 설명 & 마무리:**
{f'''에겐: 
- 각 장소마다 2-3문장으로 따뜻하고 자세한 설명 (분위기, 느낌, 추천 이유)
- 마무리: "즐거운 여행 되세요!", "편안한 여행 되시길 바라요!" 같은 따뜻한 인사''' if user_profile.personality in ["에겐남", "에겐녀"] else '''테토:
- 장소명과 핵심 정보만 간단히 (1문장 이하)
- 마무리: "끝", "이상" 또는 아예 마무리 인사 없이'''}
"""
    
    try:
        # 복잡한 일정 생성을 위한 넉넉한 타임아웃 (120초)
        response = await asyncio.wait_for(
            response_llm.ainvoke(prompt), 
            timeout=120.0
        )
        final_response = response.content.strip()
        
        # 🎯 성향별 하드코딩 후처리 (LLM이 일관성 없게 나올 때 강제 보정)
        if user_profile and user_profile.personality:
            final_response = apply_personality_hardcoding(final_response, user_profile.personality)
        
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
        print(f"❌ 개인화 응답 생성 오류: {e}")
        return {
            **state,
            "final_response": "죄송합니다. 일정 생성 중 오류가 발생했습니다.",
            "conversation_history": conversation_history
        }

def apply_personality_hardcoding(response: str, personality: str) -> str:
    """성향별 하드코딩 후처리 - 장소 설명 & 마무리 문장 성향별 처리"""
    
    # 기존 따뜻한/차가운 마무리 문구들 제거 (일관성을 위해)
    common_endings = [
        "즐거운 여행 되세요!", "편안한 여행 되시길 바라요!", "좋은 여행 되세요!",
        "마음에 드실 거예요!", "기분 좋아지실 거예요!", "힐링되실 거예요!",
        "제주에서의 즐거운 시간 되세요!", "좋은 추억 만드세요!", "안전한 여행 되세요!",
        "끝.", "이상.", "끝", "이상"
    ]
    
    for ending in common_endings:
        if response.endswith(ending):
            response = response.rstrip(ending).rstrip()
    
    # 🎯 장소 설명 처리 (테토는 장소명만, 에겐은 설명 유지)
    if personality in ["테토남", "테토녀"]:
        print(f"🔥 테토 성향 하드코딩 적용: {personality}")  # 디버깅용
        
        # 테토 성향: 무조건 강력하게 설명 제거
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 1. 일정 구조 라인들은 그대로 유지
            if (line.startswith('###') or 
                line.startswith('**오전') or line.startswith('**오후') or line.startswith('**저녁') or
                line.strip() == '' or
                '일차' in line):
                cleaned_lines.append(line)
                continue
            
            # 2. 장소 라인 처리: 콜론(:) 뒤 모든 설명 무조건 제거
            if line.startswith('- **') or line.startswith('-**'):
                # "- **장소명**: 설명" → "- **장소명**"
                # "- **장소명 (시간)**: 설명" → "- **장소명 (시간)**"
                if ':' in line:
                    # 콜론 앞부분만 가져오기
                    place_part = line.split(':')[0].strip()
                    cleaned_lines.append(place_part)
                else:
                    cleaned_lines.append(line)
                continue
            
            # 3. 기타 설명 문장들은 모두 제거 (숫자나 특수문자로 시작하지 않는 일반 문장들)
            if (line.strip() and 
                not line.startswith('#') and 
                not line.startswith('**') and 
                not line.startswith('-') and
                not line.startswith('>')):
                # 설명 문장이므로 제거
                continue
            else:
                cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines)
        print(f"🔥 테토 처리 완료. 결과 길이: {len(response)}")  # 디버깅용
    
    # 각 성향별 특징적인 마무리 문장 추가
    if personality == "에겐남":
        response += "\n\n정말 좋은 여행이 될 것 같습니다. 혹시 더 궁금한 것이 있으시면 언제든 말씀해 주세요. 편안하고 즐거운 제주 여행 되시길 바랍니다! 😊"
        
    elif personality == "에겐녀":
        response += "\n\n와~ 정말 설레는 여행 계획이네요! 혹시 걱정되는 부분이나 더 알고 싶은 게 있으시면 언제든 말씀해 주세요. 힐링 가득한 제주 여행 되시길 진심으로 바라요! 💕✨"
        
    elif personality == "테토남":
        response += "\n\n이상. 더 필요한 정보 있으면 말해."
        
    elif personality == "테토녀":
        response += "\n\n끝. 다른 거 필요하면 또 말해."
    
    return response

# 라우팅 함수들 (여행 계획 시 모든 카테고리 검색)
def should_search_hotels(state: PersonalizedGraphState) -> bool:
    """숙박 검색 필요 여부 판단 - 프로필이 있으면 무조건 검색"""
    user_profile = state.get("user_profile")
    if user_profile:
        return True  # 프로필이 있으면 무조건 검색
    
    # 프로필이 없을 때만 키워드 기반 판단
    user_message = state["user_message"].lower()
    hotel_keywords = ["숙박", "호텔", "펜션", "리조트", "게스트하우스", "잠", "머물", "체크인", "숙소"]
    return any(keyword in user_message for keyword in hotel_keywords)

def should_search_travel(state: PersonalizedGraphState) -> bool:
    """관광지 검색 필요 여부 판단 - 프로필이 있으면 무조건 검색"""
    user_profile = state.get("user_profile")
    if user_profile:
        return True  # 프로필이 있으면 무조건 검색
    
    # 프로필이 없을 때만 키워드 기반 판단
    user_message = state["user_message"].lower()
    travel_keywords = ["관광", "여행지", "명소", "가볼만한", "구경", "관광지", "장소", "코스", "여행", "일정"]
    return any(keyword in user_message for keyword in travel_keywords)

def should_search_food(state: PersonalizedGraphState) -> bool:
    """음식 검색 필요 여부 판단 - 프로필이 있으면 무조건 검색"""
    user_profile = state.get("user_profile")
    if user_profile:
        return True  # 프로필이 있으면 무조건 검색
    
    # 프로필이 없을 때만 키워드 기반 판단
    user_message = state["user_message"].lower()
    food_keywords = ["맛집", "음식", "식당", "카페", "먹을", "요리", "특산품", "디저트", "점심", "저녁", "식사"]
    return any(keyword in user_message for keyword in food_keywords)

def should_search_events(state: PersonalizedGraphState) -> bool:
    """이벤트 검색 필요 여부 판단 - 프로필이 있으면 무조건 검색"""
    user_profile = state.get("user_profile")
    if user_profile:
        return True  # 프로필이 있으면 무조건 검색
    
    # 프로필이 없을 때만 키워드 기반 판단
    user_message = state["user_message"].lower()
    event_keywords = ["축제", "이벤트", "행사", "공연", "체험", "활동", "프로그램"]
    return any(keyword in user_message for keyword in event_keywords)

# 라우팅 로직 (병렬 검색용으로 변경)
def should_continue_to_search(state: PersonalizedGraphState) -> str:
    """검색 조건 판단 - 프로필이 있으면 무조건 검색"""
    user_profile = state.get("user_profile")
    
    # 프로필이 있으면 무조건 검색
    if user_profile:
        return "parallel_search"
    else:
        return "response_generation"

# 개인화된 병렬 검색 함수
async def personalized_parallel_search_all(state: PersonalizedGraphState) -> PersonalizedGraphState:
    """모든 카테고리를 병렬로 검색 (개인화된 버전)"""
    user_profile = state["user_profile"]
    
    # 여행 기간에 따른 검색 개수 결정 (smart_chatbot과 동일)
    duration_days = 3  # 기본값
    if user_profile.duration:
        if "1박" in user_profile.duration or "1일" in user_profile.duration:
            duration_days = 1
        elif "2박" in user_profile.duration or "2일" in user_profile.duration:
            duration_days = 2
        elif "3박" in user_profile.duration or "3일" in user_profile.duration:
            duration_days = 3
        elif "4박" in user_profile.duration or "4일" in user_profile.duration:
            duration_days = 4
        elif "5박" in user_profile.duration or "5일" in user_profile.duration:
            duration_days = 5
    
    # 여행 기간별 검색 개수 (smart_chatbot.py와 완전 동일)
    search_counts = {
        1: {"hotel": 3, "tour": 4, "food": 3, "event": 2},
        2: {"hotel": 3, "tour": 6, "food": 5, "event": 3}, 
        3: {"hotel": 4, "tour": 8, "food": 6, "event": 3},
        4: {"hotel": 4, "tour": 12, "food": 8, "event": 4},
        5: {"hotel": 5, "tour": 15, "food": 10, "event": 5}
    }.get(duration_days, {"hotel": 5, "tour": 18, "food": 12, "event": 6})
    
    print(f"📊 여행 기간 {duration_days}일 기준 검색 개수: {search_counts}")
    
    print(f"🔍 개인화된 병렬 검색 시작 - 검색 개수: {search_counts}")
    
    # 각 카테고리별 개인화된 쿼리 생성 및 병렬 검색
    async def search_hotels():
        should_search = should_search_hotels(state)
        print(f"🏨 호텔 검색 조건: {should_search}")
        
        if should_search:
            # 여행 스타일 기반 강화된 숙박 쿼리 생성
            accommodation_style = ""
            if user_profile.travel_style:
                style = user_profile.travel_style.lower()
                if "액티비티" in style or "활동" in style or "모험" in style:
                    accommodation_style = "액티비티 중심의 편리한 위치의"
                elif "힐링" in style or "휴식" in style or "여유" in style:
                    accommodation_style = "힐링과 휴식을 위한 조용하고 평화로운"
                elif "감성" in style or "카페" in style or "예쁜" in style:
                    accommodation_style = "감성적이고 분위기 좋은"
                elif "음식" in style or "맛집" in style:
                    accommodation_style = "맛집 접근성이 좋은"
                elif "자연" in style or "풍경" in style:
                    accommodation_style = "자연 풍경이 아름다운"
                else:
                    accommodation_style = "편안한"
            
            # 성향별 숙박 특성
            personality_feature = ""
            if user_profile.personality in ["테토남", "테토녀"]:
                personality_feature = "효율적이고 모던한"
            elif user_profile.personality in ["에겐남", "에겐녀"]:
                personality_feature = "따뜻하고 아늑한"
            
            # 동행자별 숙박 특성
            group_feature = ""
            if user_profile.group_type == "커플":
                group_feature = "로맨틱한 오션뷰"
            elif user_profile.group_type == "가족":
                group_feature = "가족 친화적인"
            elif user_profile.group_type == "친구":
                group_feature = "넓고 편리한"
            elif user_profile.group_type == "혼자":
                group_feature = "1인 여행객에게 최적인"
            
            region_part = f"{user_profile.travel_region or '제주도'} 지역"
            style_part = f"{accommodation_style} {personality_feature} {group_feature}".strip()
            
            query = f"{region_part}의 {style_part} 호텔과 숙박시설을 추천해주세요"
            print(f"🏨 호텔 검색 쿼리: '{query}' (개수: {search_counts['hotel']})")
            
            results = await search_vector_db(query, "hotel", top_k=search_counts["hotel"])
            print(f"🏨 호텔 검색 결과: {len(results)}개")
            return results
        return []
    
    async def search_tours():
        should_search = should_search_travel(state)
        print(f"🗺️ 관광지 검색 조건: {should_search}")
        
        if should_search:
            # 여행 스타일 기반 강화된 쿼리 생성
            interests = " ".join(user_profile.interests) if user_profile.interests else "관광"
            
            # 여행 스타일에서 핵심 키워드 추출
            travel_style_keywords = ""
            if user_profile.travel_style:
                style = user_profile.travel_style.lower()
                if "액티비티" in style or "활동" in style or "모험" in style:
                    travel_style_keywords = "액티비티와 모험을 즐길 수 있는 스릴넘치는"
                elif "힐링" in style or "휴식" in style or "여유" in style:
                    travel_style_keywords = "힐링과 휴식을 위한 평화로운"  
                elif "감성" in style or "카페" in style or "예쁜" in style:
                    travel_style_keywords = "감성적이고 예쁜 분위기의"
                elif "음식" in style or "맛집" in style:
                    travel_style_keywords = "맛집과 연계된"
                elif "자연" in style or "풍경" in style:
                    travel_style_keywords = "아름다운 자연풍경의"
                else:
                    travel_style_keywords = user_profile.travel_style.replace("여행", "").strip()
            
            # 성향별 형용사 추가
            personality_adj = ""
            if user_profile.personality in ["테토남", "테토녀"]:
                personality_adj = "도전적이고 특별한"
            elif user_profile.personality in ["에겐남", "에겐녀"]:
                personality_adj = "편안하고 따뜻한"
            
            # 통합 쿼리 생성
            region_part = f"{user_profile.travel_region or '제주도'} 지역"
            group_part = f"{user_profile.group_type or '여행객'}"
            style_part = f"{travel_style_keywords} {personality_adj}".strip()
            
            query = f"{region_part}에서 {group_part}이 {interests}을 즐길 수 있는 {style_part} 관광지와 명소를 찾아주세요"
            print(f"🗺️ 관광지 검색 쿼리: '{query}' (개수: {search_counts['tour']})")
            
            results = await search_vector_db(query, "travel", top_k=search_counts["tour"])
            print(f"🗺️ 관광지 검색 결과: {len(results)}개")
            return results
        return []
    
    async def search_foods():
        should_search = should_search_food(state)
        print(f"🍽️ 음식 검색 조건: {should_search}")
        
        if should_search:
            # 여행 스타일 기반 강화된 음식 쿼리 생성
            food_style = ""
            if user_profile.travel_style:
                style = user_profile.travel_style.lower()
                if "액티비티" in style or "활동" in style or "모험" in style:
                    food_style = "에너지 충전을 위한 든든한"
                elif "힐링" in style or "휴식" in style or "여유" in style:
                    food_style = "힐링되는 편안한 분위기의"
                elif "감성" in style or "카페" in style or "예쁜" in style:
                    food_style = "감성적이고 분위기 좋은"
                elif "음식" in style or "맛집" in style:
                    food_style = "현지인이 인정하는 진짜"
                elif "자연" in style or "풍경" in style:
                    food_style = "자연과 함께하는 뷰 맛집"
                else:
                    food_style = "맛있는"
            
            # 성향별 음식 특성
            personality_food = ""
            if user_profile.personality in ["테토남", "테토녀"]:
                personality_food = "유명한 핫플레이스"
            elif user_profile.personality in ["에겐남", "에겐녀"]:
                personality_food = "따뜻하고 정겨운"
            
            # 동행자별 음식점 특성
            group_food = ""
            if user_profile.group_type == "커플":
                group_food = "로맨틱한 데이트"
            elif user_profile.group_type == "가족":
                group_food = "가족 단위로 즐기기 좋은"
            elif user_profile.group_type == "친구":
                group_food = "친구들과 함께 가기 좋은"
            elif user_profile.group_type == "혼자":
                group_food = "혼밥하기 좋은"
            
            region_part = f"{user_profile.travel_region or '제주도'} 지역"
            style_part = f"{food_style} {personality_food} {group_food}".strip()
            
            query = f"{region_part}의 {style_part} 맛집과 식당을 추천해주세요"
            print(f"🍽️ 음식 검색 쿼리: '{query}' (개수: {search_counts['food']})")
            
            results = await search_vector_db(query, "food", top_k=search_counts["food"])
            print(f"🍽️ 음식 검색 결과: {len(results)}개")
            return results
        return []
    
    async def search_events():
        should_search = should_search_events(state)
        print(f"🎉 이벤트 검색 조건: {should_search}")
        
        if should_search:
            # 여행 스타일 기반 강화된 이벤트 쿼리 생성
            event_style = ""
            if user_profile.travel_style:
                style = user_profile.travel_style.lower()
                if "액티비티" in style or "활동" in style or "모험" in style:
                    event_style = "액티비티와 체험 활동 중심의 역동적인"
                elif "힐링" in style or "휴식" in style or "여유" in style:
                    event_style = "힐링과 여유를 느낄 수 있는 평화로운"
                elif "감성" in style or "카페" in style or "예쁜" in style:
                    event_style = "감성적이고 포토제닉한"
                elif "음식" in style or "맛집" in style:
                    event_style = "음식과 관련된"
                elif "자연" in style or "풍경" in style:
                    event_style = "자연과 함께하는"
                else:
                    event_style = "재미있는"
            
            # 성향별 이벤트 특성
            personality_event = ""
            if user_profile.personality in ["테토남", "테토녀"]:
                personality_event = "인기 있고 트렌디한"
            elif user_profile.personality in ["에겐남", "에겐녀"]:
                personality_event = "따뜻한 분위기의"
            
            # 동행자별 이벤트 특성
            group_event = ""
            if user_profile.group_type == "커플":
                group_event = "커플이 함께 즐기기 좋은"
            elif user_profile.group_type == "가족":
                group_event = "가족 단위로 참여하기 좋은"
            elif user_profile.group_type == "친구":
                group_event = "친구들과 함께 즐기기 좋은"
            elif user_profile.group_type == "혼자":
                group_event = "혼자서도 즐길 수 있는"
            
            region_part = f"{user_profile.travel_region or '제주도'} 지역"
            style_part = f"{event_style} {personality_event} {group_event}".strip()
            
            query = f"{region_part}의 {style_part} 이벤트와 축제, 체험 활동을 추천해주세요"
            print(f"🎉 이벤트 검색 쿼리: '{query}' (개수: {search_counts['event']})")
            
            results = await search_vector_db(query, "event", top_k=search_counts["event"])
            print(f"🎉 이벤트 검색 결과: {len(results)}개")
            return results
        return []
    
    # 병렬 실행
    try:
        hotel_results, tour_results, food_results, event_results = await asyncio.gather(
            search_hotels(),
            search_tours(), 
            search_foods(),
            search_events(),
            return_exceptions=True
        )
        
        # 예외 처리 및 디버깅
        if isinstance(hotel_results, Exception):
            print(f"❌ 호텔 검색 오류: {hotel_results}")
            hotel_results = []
        if isinstance(tour_results, Exception):
            print(f"❌ 관광지 검색 오류: {tour_results}")
            tour_results = []
        if isinstance(food_results, Exception):
            print(f"❌ 음식 검색 오류: {food_results}")
            food_results = []
        if isinstance(event_results, Exception):
            print(f"❌ 이벤트 검색 오류: {event_results}")
            event_results = []
        
        print(f"✅ 개인화된 병렬 검색 완료 - 호텔: {len(hotel_results)}, 관광: {len(tour_results)}, 음식: {len(food_results)}, 이벤트: {len(event_results)}")
        
        return {
            "hotel_results": hotel_results,
            "travel_results": tour_results,
            "food_results": food_results,
            "event_results": event_results
        }
        
    except Exception as e:
        print(f"❌ 개인화된 병렬 검색 오류: {e}")
        return {
            "hotel_results": [],
            "travel_results": [],
            "food_results": [],
            "event_results": []
        }

# 라우팅 함수들 (smart_chatbot.py와 동일)
def should_continue_to_agents(state: PersonalizedGraphState) -> str:
    """프로필이 준비되었는지 확인하여 다음 단계 결정"""
    if state.get("profile_ready", False):
        return "parallel_search"  # 병렬 검색으로 변경
    else:
        return "end"

def should_continue_to_response(state: PersonalizedGraphState) -> str:
    """병렬 검색 후 응답 생성으로 이동"""
    return "response_generation"

# 그래프 구성 (smart_chatbot.py와 동일한 구조)
def create_personalized_graph():
    """개인화된 그래프 생성"""
    workflow = StateGraph(PersonalizedGraphState)
    
    # 노드 추가
    workflow.add_node("profile_collection", personalized_profile_collector_node)
    workflow.add_node("parallel_search", personalized_parallel_search_all)  # 병렬 검색 노드 추가
    workflow.add_node("response_generation", personalized_response_node)
    
    # 기존 개별 에이전트들은 유지 (필요시 사용)
    workflow.add_node("hotel_agent", hotel_agent_node)
    workflow.add_node("travel_agent", travel_agent_node)
    workflow.add_node("food_agent", food_agent_node)
    workflow.add_node("event_agent", event_agent_node)
    
    # 시작점 설정
    workflow.set_entry_point("profile_collection")
    
    # 조건부 라우팅 (smart_chatbot.py와 동일)
    workflow.add_conditional_edges(
        "profile_collection",
        should_continue_to_agents,
        {
            "parallel_search": "parallel_search",
            "end": END
        }
    )
    
    # 병렬 검색 → 응답 생성
    workflow.add_conditional_edges(
        "parallel_search",
        should_continue_to_response,
        {
            "response_generation": "response_generation"
        }
    )
    
    # 종료점 설정
    workflow.add_edge("response_generation", END)
    
    # 메모리 설정
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

# 그래프 생성
graph = create_personalized_graph()

# 메인 개인화 챗봇 클래스 (smart_chatbot.py와 동일 구조)
class PersonalizedJejuChatbot:
    """LangGraph 기반 개인화된 멀티 에이전트 제주도 여행 챗봇"""
    
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
                        "user_profile": PersonalizedUserProfile(),
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
                    "user_profile": PersonalizedUserProfile(),
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
            user_profile = result.get("user_profile", PersonalizedUserProfile())
            
            return {
                "response": response_text,
                "user_profile": user_profile
            }
            
        except Exception as e:
            print(f"❌ 개인화 챗봇 실행 오류: {e}")
            return {
                "response": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
                "user_profile": PersonalizedUserProfile()
            }

# FastAPI 서버 코드는 별도 파일로 분리 예정
if __name__ == "__main__":
    print("개인화된 제주도 여행 챗봇이 준비되었습니다!")
    print(f"로딩된 사용자 데이터: {len(USER_DATA)}명")