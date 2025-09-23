"""
🔗 LangChain 기반 제주도 여행 단일 체인 시스템 (비교군)
- 순차적 정보 처리
- 단일 LLM으로 모든 카테고리 처리
- 기존 RAG 패턴 구현
- 멀티에이전트 vs 단일체인 성능 비교용
"""

import asyncio
import httpx
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# 환경변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 글로벌 로그 수집기
execution_logs = []

def add_log(message: str, log_type: str = "info"):
    """실행 로그 추가"""
    global execution_logs
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        "timestamp": timestamp,
        "type": log_type,
        "message": message
    }
    execution_logs.append(log_entry)
    print(message)  # 기존 콘솔 출력 유지

def clear_logs():
    """로그 초기화"""
    global execution_logs
    execution_logs = []

def get_logs():
    """현재 로그 반환"""
    global execution_logs
    return execution_logs.copy()

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

# 단일 LLM 인스턴스 (모든 작업에 공용)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4-turbo", temperature=0.7)

# 벡터 DB 접근 URL
RAG_URL = "http://localhost:8002/chat"
RAG_SEARCH_URL = "http://localhost:8002/search"  # 검색 전용 엔드포인트

# 글로벌 HTTP 클라이언트 (연결 재사용)
global_http_client = None

async def get_global_client():
    """글로벌 HTTP 클라이언트 가져오기 (연결 재사용)"""
    global global_http_client
    if global_http_client is None:
        timeout_config = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
        global_http_client = httpx.AsyncClient(
            timeout=timeout_config, 
            limits=limits,
            http2=True  # HTTP/2 지원
        )
    return global_http_client

class ProfileExtractorParser(BaseOutputParser):
    """프로필 정보 추출 파서"""
    
    def parse(self, text: str) -> Dict:
        try:
            # JSON 추출
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].strip()
            return json.loads(text)
        except Exception as e:
            print(f"❌ 프로필 파싱 오류: {e}")
            return {}

# 체인 1: 프로필 수집 체인
profile_extraction_prompt = PromptTemplate(
    input_variables=["message", "current_profile"],
    template="""다음 사용자 메시지에서 제주도 여행 관련 정보를 추출해주세요.

사용자 메시지: {message}

현재 프로필: {current_profile}

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
)

profile_chain = LLMChain(
    llm=llm,
    prompt=profile_extraction_prompt,
    output_parser=ProfileExtractorParser()
)

# 체인 2: 정보 수집 응답 체인
info_collection_prompt = PromptTemplate(
    input_variables=["profile_summary", "user_message", "history_context"],
    template="""제주도 여행 상담사로서 사용자와 자연스럽게 대화하면서 필요한 정보를 수집해주세요.

현재 수집된 정보: {profile_summary}
사용자 최신 메시지: {user_message}{history_context}

**응답 가이드:**
- 이미 언급된 정보는 다시 묻지 않기
- 부족한 핵심 정보(여행 기간, 여행 유형, 관심사 등) 자연스럽게 확인
- 강요하지 않고 대화 맥락에 맞게 정보 수집
- 친근하고 도움되는 톤 유지
- 현재 정보로도 추천 가능함을 안내"""
)

info_collection_chain = LLMChain(llm=llm, prompt=info_collection_prompt)

# 체인 3: 통합 검색 쿼리 생성 체인
search_query_prompt = PromptTemplate(
    input_variables=["profile_summary", "category"],
    template="""당신은 제주도 여행 검색 쿼리 생성 전문가입니다.

사용자 프로필: {profile_summary}
검색 카테고리: {category}

다음 카테고리에 맞는 벡터 DB 검색 쿼리를 생성해주세요:

- hotel: 숙박시설 (호텔, 펜션, 리조트 등)
- travel: 관광지 (명소, 체험, 액티비티 등)  
- food: 식당/카페 (맛집, 음식점, 카페 등)
- event: 행사/이벤트 (축제, 문화행사 등)

사용자의 여행 유형, 관심사, 지역을 고려하여 자연어 검색 쿼리를 한 줄로 생성해주세요.

검색 쿼리:"""
)

search_query_chain = LLMChain(llm=llm, prompt=search_query_prompt)

# 체인 4: 최종 일정 생성 체인
itinerary_generation_prompt = PromptTemplate(
    input_variables=["profile_summary", "history_summary", "hotel_data", "travel_data", "food_data", "event_data"],
    template="""
[시스템 메시지]
당신은 제주 여행 일정 추천 전문가 '오르미'입니다.

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

[실제 태스크]
아래 정보를 바탕으로, 위 형식대로 제주도 일정을 구성하세요.

**입력 정보:**
- 사용자 프로필: {profile_summary}
- 최근 대화 내용: {history_summary}
- 숙박 정보: {hotel_data}
- 관광 정보: {travel_data}  
- 음식 정보: {food_data}
- 이벤트 정보: {event_data}

**작성 지침:**
- 사용자 성향과 대화 맥락을 반영해 **개인화된 일정**을 작성하세요.
- 시간대별로 **1~2개 장소**를 추천하며, **아침/점심/저녁 식사 장소는 반드시 포함**하세요.
- **관광 목적의 카페는 하루 1개까지만** 포함하세요.
- **장소 설명은 제공된 정보만 사용**하고, 추측은 절대 하지 마세요.
- **모든 장소는 정확한 이름과 주소를 반드시 포함**하여 작성하세요.
- **1일차 오후에 숙소 체크인**, 모든 날은 **숙소에서 마무리**, 마지막 날은 **공항에서 마무리**되도록 하세요.
"""
)

itinerary_chain = LLMChain(llm=llm, prompt=itinerary_generation_prompt)

class SingleChainJejuChatbot:
    """LangChain 기반 단일 체인 제주도 여행 챗봇 (비교군)"""
    
    def __init__(self):
        self.conversation_history = []
        self.user_profile = UserProfile()
        self.session_id = "default"
        self.llm = llm  # LLM 인스턴스 추가
    
    def is_profile_sufficient(self, profile: UserProfile) -> bool:
        """프로필이 충분한지 확인"""
        required_info_count = sum([
            bool(profile.travel_dates),
            bool(profile.duration),
            bool(profile.group_type),
            bool(profile.interests),
            bool(profile.budget),
            bool(profile.travel_region)
        ])
        
        result = required_info_count >= 3
        print(f"🧪 프로필 충분성 판단: {result} (필요정보: {required_info_count}/6)")
        return result
    
    def update_profile(self, profile_info: Dict) -> None:
        """프로필 업데이트"""
        if profile_info.get("travel_dates"):
            self.user_profile.travel_dates = profile_info["travel_dates"]
        if profile_info.get("duration"):
            self.user_profile.duration = profile_info["duration"]
        if profile_info.get("group_type"):
            self.user_profile.group_type = profile_info["group_type"]
        if profile_info.get("interests"):
            new_interests = profile_info["interests"]
            for interest in new_interests:
                if interest not in self.user_profile.interests:
                    self.user_profile.interests.append(interest)
        if profile_info.get("budget"):
            self.user_profile.budget = profile_info["budget"]
        if profile_info.get("travel_region"):
            self.user_profile.travel_region = profile_info["travel_region"]
    
    def extract_profile_simple(self, message: str) -> Dict:
        """간단한 키워드 매칭으로 프로필 추출 (LLM 호출 없음)"""
        message_lower = message.lower()
        profile_info = {}
        
        # 여행 기간 추출
        if "1박" in message_lower or "2일" in message_lower:
            profile_info["duration"] = "1박2일"
        elif "2박" in message_lower or "3일" in message_lower:
            profile_info["duration"] = "2박3일"
        elif "3박" in message_lower or "4일" in message_lower:
            profile_info["duration"] = "3박4일"
        
        # 여행 유형 추출
        if any(word in message_lower for word in ["커플", "연인", "남친", "여친", "애인"]):
            profile_info["group_type"] = "커플"
        elif any(word in message_lower for word in ["가족", "아이", "부모", "엄마", "아빠"]):
            profile_info["group_type"] = "가족"
        elif any(word in message_lower for word in ["친구", "동료", "같이"]):
            profile_info["group_type"] = "친구"
        elif any(word in message_lower for word in ["혼자", "혼행", "솔로"]):
            profile_info["group_type"] = "혼자"
        
        # 관심사 추출
        interests = []
        if any(word in message_lower for word in ["맛집", "음식", "먹거리"]):
            interests.append("맛집")
        if any(word in message_lower for word in ["힐링", "휴식", "쉬고"]):
            interests.append("힐링")
        if any(word in message_lower for word in ["액티비티", "체험", "활동"]):
            interests.append("액티비티")
        if any(word in message_lower for word in ["사진", "인스타", "감성"]):
            interests.append("사진촬영")
        if interests:
            profile_info["interests"] = interests
        
        # 지역 추출
        if any(word in message_lower for word in ["제주시", "제주 시내", "공항 근처"]):
            profile_info["travel_region"] = "제주시"
        elif any(word in message_lower for word in ["서귀포", "중문", "성산"]):
            profile_info["travel_region"] = "서귀포"
        elif any(word in message_lower for word in ["서쪽", "한림", "협재"]):
            profile_info["travel_region"] = "제주 서쪽"
        elif any(word in message_lower for word in ["동쪽", "성산일출봉"]):
            profile_info["travel_region"] = "제주 동쪽"
        
        return profile_info
    
    def calculate_search_counts(self, duration: str) -> Dict[str, int]:
        """여행 기간에 따라 카테고리별 검색 개수 결정 (멀티에이전트와 동일)"""
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
        
        # 일수별 검색 개수 설정 (멀티에이전트와 동일)
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
    
    async def search_vector_db(self, query: str, category: str = "", top_k: int = 8) -> List[Dict]:
        """벡터 DB 검색 (상세 시간 분석 포함)"""
        import asyncio
        total_start = asyncio.get_event_loop().time()
        
        try:
            # 1. 전역 클라이언트 가져오기 (연결 재사용)
            setup_start = asyncio.get_event_loop().time()
            client = await get_global_client()
            setup_time = asyncio.get_event_loop().time() - setup_start
            
            # 2. 페이로드 준비 시간
            payload_start = asyncio.get_event_loop().time()
            search_payload = {
                "query": query,
                "top_k": top_k,
                "search_type": "similarity",
                "filters": {}
            }
            payload_time = asyncio.get_event_loop().time() - payload_start
            
            # 3. HTTP 요청 시간 (연결 재사용으로 빨라짐)
            request_start = asyncio.get_event_loop().time()
            print(f"🔄 {category} 요청 시작: {query[:30]}...")
            
            response = await client.post(RAG_SEARCH_URL, json=search_payload)
            
            request_time = asyncio.get_event_loop().time() - request_start
            
            # 4. 응답 처리 시간
            process_start = asyncio.get_event_loop().time()
            
            if response.status_code == 200:
                result = response.json()
                sources = result.get("results", [])  # /search 엔드포인트는 'results' 키를 사용
                processing_time = result.get("processing_time", 0)
                
                process_time = asyncio.get_event_loop().time() - process_start
                total_time = asyncio.get_event_loop().time() - total_start
                
                print(f"✅ {category} 완료 - {len(sources)}개 결과")
                print(f"   📊 시간 분석: 클라이언트({setup_time*1000:.1f}ms) + 페이로드({payload_time*1000:.1f}ms) + 요청({request_time:.2f}s) + 처리({process_time*1000:.1f}ms) = 총({total_time:.2f}s)")
                print(f"   🔍 RAG 서버 처리: {processing_time:.2f}초")
                print(f"   🚀 네트워크 최적화: 연결 재사용")
                
                return sources[:top_k]
            else:
                total_time = asyncio.get_event_loop().time() - total_start
                print(f"❌ {category} HTTP 오류 - 상태코드: {response.status_code}, 총 시간: {total_time:.2f}초")
                return []
                
        except Exception as e:
            total_time = asyncio.get_event_loop().time() - total_start
            print(f"❌ {category} 검색 오류: {e}, 총 시간: {total_time:.2f}초")
            return []
    
    async def direct_search_all_categories(self, profile: UserProfile) -> Dict:
        """쿼리 재생성 없이 바로 순차 검색 (공정한 비교를 위해)"""
        # 로그 초기화
        clear_logs()
        
        add_log("🔗 단일 체인 - 사전 정의된 쿼리로 바로 순차 검색 시작...", "single_chain")
        
        # LLM 호출 횟수 초기화 (프로필 추출 1회 이미 완료된 상태)
        llm_calls_count = 1  # 프로필 추출에서 1회 사용
        add_log(f"📊 프로필 추출에서 LLM 호출: 1회", "single_chain")
        
        categories = ["hotel", "tour", "food", "event"]
        # 멀티에이전트와 동일한 동적 검색 개수 사용
        search_counts = self.calculate_search_counts(profile.duration)
        
        # 진정한 순차 처리: 쿼리생성 → 검색 → 쿼리생성 → 검색 반복
        add_log("🔗 단일에이전트 순차 처리 시작 - 쿼리생성 → 검색 반복", "single_chain")
        
        profile_summary = profile.get_summary()
        
        # 🕐 전체 검색 시간 측정 시작 (첫 쿼리생성 시작 순간)
        search_start_time = asyncio.get_event_loop().time()
        add_log(f"🔍 순차 처리 시작! (시작 시간: {search_start_time:.3f})", "single_chain")
        
        all_results = {}
        queries = {}
        total_query_generation_time = 0
        category_timings = {}  # 각 카테고리별 상세 시간 저장
        
        for category in categories:
            try:
                # 1단계: 개별 쿼리 생성
                add_log(f"🧠 {category} 쿼리 생성 시작...", "sequential")
                
                if category == "hotel":
                    prompt = f"""당신은 제주 여행자를 위한 **숙박 검색 쿼리 생성 전문가**입니다.
사용자 프로필: {profile_summary}
벡터 DB에서 숙박을 검색하기 위한 자연어 검색 쿼리 문장 한 줄을 생성해주세요.
"제주도", "숙박", "호텔" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.
검색 쿼리:"""
                elif category == "tour":
                    prompt = f"""당신은 제주관광 전문 **자연어** **쿼리 생성 전문가**입니다.
사용자 프로필: {profile_summary}
벡터 DB에서 관광지 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄을 만들어주세요.
"제주도", "관광지" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.
검색 쿼리:"""
                elif category == "food":
                    prompt = f"""당신은 제주관광 전문 **자연어 쿼리 생성 전문가**입니다.
사용자 프로필: {profile_summary}
벡터 DB에서 식당 또는 카페 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄을 만들어주세요.
"제주도", "맛집" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.
검색 쿼리:"""
                elif category == "event":
                    prompt = f"""당신은 제주관광 전문 **자연어 쿼리 생성 전문가**입니다.
사용자 프로필: {profile_summary}
벡터 DB에서 이벤트나 축제 정보를 검색하기 위한 자연어 검색 쿼리 문장 한 줄을 만들어주세요.
"제주도", "이벤트", "축제" 등 핵심 키워드를 포함하고 자연스럽고 간결해야 합니다.
검색 쿼리:"""
                
                query_start = asyncio.get_event_loop().time()
                response = await self.llm.ainvoke(prompt)
                search_query = response.content.strip()
                query_time = asyncio.get_event_loop().time() - query_start
                total_query_generation_time += query_time
                
                # LLM 호출 횟수 증가
                llm_calls_count += 1
                
                queries[category] = search_query
                add_log(f"🧠 {category} 쿼리 생성 완료 ({query_time:.2f}초): '{search_query}'", "sequential")
                add_log(f"📊 {category} 쿼리 생성 LLM 호출: +1회 (총 {llm_calls_count}회)", "sequential")
                
                # 2단계: 즉시 검색 실행
                add_log(f"🔍 {category} 검색 시작...", "sequential")
                search_start = asyncio.get_event_loop().time()
                search_results = await self.search_vector_db(search_query, category, search_counts[category])
                search_time = asyncio.get_event_loop().time() - search_start
                all_results[category] = search_results
                
                # 카테고리별 상세 시간 저장
                category_timings[category] = {
                    "query_generation_time": query_time,
                    "search_time": search_time,
                    "total_time": query_time + search_time
                }
                
                add_log(f"✅ {category} 완료: {len(search_results)}개 결과", "sequential")
                add_log(f"   📊 {category} 시간: 쿼리생성({query_time:.2f}초) + 검색({search_time:.2f}초) = 총({query_time + search_time:.2f}초)", "timing")
                
                # 각 검색 사이 대기 (순차 처리 특성 유지)
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"❌ {category} 처리 실패: {e}")
                all_results[category] = []
                queries[category] = ""
                category_timings[category] = {
                    "query_generation_time": 0.0,
                    "search_time": 0.0,
                    "total_time": 0.0
                }
        
        # 🕐 전체 검색 시간 측정 완료 (모든 검색 완료 후, 응답 생성 전)
        search_end_time = asyncio.get_event_loop().time()
        search_duration = search_end_time - search_start_time
        
        # 순차 검색 완료 후 결과 요약
        total_results = sum(len(all_results.get(cat, [])) for cat in ["hotel", "tour", "food", "event"])
        add_log(f"🎉 모든 검색 완료! 총 {total_results}개 결과 수집", "single_chain")
        add_log(f"⏱️  전체 검색 시간: {search_duration:.2f}초 (단일체인 순차)", "single_chain")
        
        # 검색 시간과 쿼리 정보를 결과에 포함
        # 카테고리별 상세 시간 출력
        add_log("📊 카테고리별 상세 시간 분석:", "timing")
        for category, timing in category_timings.items():
            add_log(f"   {category}: 쿼리생성({timing['query_generation_time']:.2f}초) + 검색({timing['search_time']:.2f}초) = 총({timing['total_time']:.2f}초)", "timing")
        
        # 최종 LLM 호출 횟수 로그
        add_log(f"📊 총 LLM 호출 횟수: {llm_calls_count}회 (프로필 1회 + 쿼리생성 4회)", "single_chain")
        
        return {
            "results": all_results,
            "search_duration": search_duration,
            "search_queries": queries,  # 사용된 쿼리들
            "timing_details": {  # 상세 시간 분석 (순차 처리)
                "query_generation_time": total_query_generation_time,  # 총 쿼리 생성 시간
                "sequential_execution_time": search_duration,
                "total_search_time": search_duration,
                "category_timings": category_timings  # 각 카테고리별 상세 시간
            },
            "execution_logs": get_logs(),  # 실행 로그 추가
            "llm_calls_count": llm_calls_count  # LLM 호출 횟수 추가
        }
    
    async def chat(self, user_message: str) -> Dict:
        """메인 채팅 처리 (단일 체인 구조)"""
        start_time = datetime.now()
        print(f"🔗 단일 체인 처리 시작: {start_time.strftime('%H:%M:%S')}")
        
        try:
            # 대화 기록에 사용자 메시지 추가
            self.conversation_history.append({
                "role": "user",
                "message": user_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # 1단계: 프로필 정보 추출 (체인 1)
            print("📋 1단계: 프로필 정보 추출")
            profile_info = await profile_chain.arun(
                message=user_message,
                current_profile=self.user_profile.get_summary()
            )
            print(f"🔍 추출된 프로필: {profile_info}")
            
            # 프로필 업데이트
            self.update_profile(profile_info)
            print(f"📝 업데이트된 프로필: {self.user_profile.get_summary()}")
            
            # 2단계: 프로필 충분성 확인
            if not self.is_profile_sufficient(self.user_profile):
                print("📋 2단계: 추가 정보 수집")
                
                # 대화 히스토리 컨텍스트
                history_context = ""
                if len(self.conversation_history) > 1:
                    recent_messages = [msg for msg in self.conversation_history[-4:] if msg['role'] == 'user']
                    if recent_messages:
                        history_context = f"\n이전 대화: {', '.join([msg['message'][:50] for msg in recent_messages])}"
                
                # 정보 수집 응답 생성 (체인 2)
                response = await info_collection_chain.arun(
                    profile_summary=self.user_profile.get_summary(),
                    user_message=user_message,
                    history_context=history_context
                )
                
                self.conversation_history.append({
                    "role": "assistant",
                    "message": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                print(f"⏱️ 단일 체인 처리 완료: {processing_time:.2f}초 (정보 수집)")
                
                return {
                    "response": response,
                    "user_profile": self.user_profile,
                    "processing_time": processing_time
                }
            
            # 3단계: 사전 정의된 쿼리로 바로 순차 검색
            print("📋 3단계: 사전 정의된 쿼리로 바로 순차 검색 (단일 체인)")
            search_response = await self.direct_search_all_categories(self.user_profile)
            search_results = search_response["results"]
            search_duration = search_response["search_duration"]
            search_queries = search_response["search_queries"]
            timing_details = search_response["timing_details"]
            
            # 4단계: 최종 일정 생성 (체인 4)
            print("📋 4단계: 최종 일정 생성")
            
            # 대화 히스토리 요약
            history_summary = ""
            if self.conversation_history:
                recent_messages = self.conversation_history[-6:]
                history_summary = "\n".join([f"- {msg['role']}: {msg['message'][:100]}{'...' if len(msg['message']) > 100 else ''}" for msg in recent_messages])
            
            # 검색 결과를 JSON 문자열로 변환
            hotel_data = json.dumps([{"name": h.get("name", ""), "address": h.get("address", ""), "description": str(h.get("content") or h.get("description") or "")} for h in search_results.get("hotel", [])], ensure_ascii=False)
            travel_data = json.dumps([{"name": t.get("name", ""), "address": t.get("address", ""), "description": str(t.get("content") or t.get("description") or "")} for t in search_results.get("tour", [])], ensure_ascii=False)
            food_data = json.dumps([{"name": f.get("name", ""), "address": f.get("address", ""), "description": str(f.get("content") or f.get("description") or "")} for f in search_results.get("food", [])], ensure_ascii=False)
            event_data = json.dumps([{"name": e.get("name", ""), "address": e.get("address", ""), "description": str(e.get("content") or e.get("description") or "")} for e in search_results.get("event", [])], ensure_ascii=False)
            
            # 최종 일정 생성
            final_response = await itinerary_chain.arun(
                profile_summary=self.user_profile.get_summary(),
                history_summary=history_summary or "첫 질문입니다",
                hotel_data=hotel_data,
                travel_data=travel_data,
                food_data=food_data,
                event_data=event_data
            )
            
            # 최종 응답 생성 LLM 호출 +1
            current_llm_calls = search_response.get("llm_calls_count", 5)  # 프로필 1회 + 쿼리생성 4회
            final_llm_calls = current_llm_calls + 1
            print(f"📊 최종 응답 생성 LLM 호출: +1회")
            print(f"📊 단일체인 총 LLM 호출 횟수: {final_llm_calls}회")
            
            # 대화 기록에 응답 추가
            self.conversation_history.append({
                "role": "assistant",
                "message": final_response,
                "timestamp": datetime.now().isoformat()
            })
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            print(f"⏱️ 단일 체인 전체 처리 완료: {processing_time:.2f}초")
            
            return {
                "response": final_response,
                "user_profile": self.user_profile,
                "processing_time": processing_time,
                "search_results": search_results,
                "search_duration": search_duration,  # 순수 검색 시간 추가
                "search_queries": search_queries,  # 사용된 쿼리들
                "timing_details": timing_details,  # 상세 시간 분석
                "execution_logs": search_response.get("execution_logs", []),  # 실행 로그 추가
                "llm_calls_count": final_llm_calls  # 총 LLM 호출 횟수
            }
            
        except Exception as e:
            print(f"❌ 단일 체인 처리 오류: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "response": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
                "user_profile": self.user_profile,
                "processing_time": processing_time,
                "search_duration": 0.0,  # 오류 시 검색 시간 0
                "llm_calls_count": 0  # 오류 시 LLM 호출 횟수 0
            }

# FastAPI 서버 설정
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="🔗 LangChain 단일 체인 제주도 챗봇 (비교군)")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 챗봇 인스턴스
chatbot = SingleChainJejuChatbot()

class ChatRequest(BaseModel):
    content: str
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
    processing_time: Optional[float] = None
    search_duration: Optional[float] = None  # 순수 검색 시간 추가
    search_queries: Optional[Dict] = None  # 사용된 검색 쿼리들
    timing_details: Optional[Dict] = None  # 상세 시간 분석
    execution_logs: Optional[List[Dict]] = None  # 실행 로그
    llm_calls_count: Optional[int] = None  # LLM 호출 횟수

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """단일 체인 채팅 엔드포인트"""
    
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
            timestamp=datetime.now().isoformat(),
            processing_time=result.get("processing_time"),
            search_duration=result.get("search_duration", 0.0),  # 검색 시간 추가
            search_queries=result.get("search_queries", {}),  # 사용된 쿼리들 추가
            timing_details=result.get("timing_details", {}),  # 상세 시간 분석 추가
            execution_logs=result.get("execution_logs", []),  # 실행 로그 추가
            llm_calls_count=result.get("llm_calls_count", 0)  # LLM 호출 횟수 추가
        )
        
    except Exception as e:
        print(f"❌ 단일 체인 채팅 오류: {e}")
        return ChatResponse(
            response="죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
            session_id=request.session_id or "default",
            needs_more_info=True,
            profile_completion=0.0,
            follow_up_questions=[],
            user_profile={},
            analysis_confidence=0.0,
            timestamp=datetime.now().isoformat(),
            processing_time=0.0,
            search_duration=0.0  # 오류 시 검색 시간 0
        )

@app.get("/")
async def root():
    return {"message": "🔗 LangChain 기반 단일 체인 제주도 챗봇 (비교군)"}

@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {
        "chatbot_status": "healthy",
        "architecture": "single_chain",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "sequential_processing": True,
            "single_llm": True,
            "langchain_based": True,
            "comparison_baseline": True
        }
    }

@app.get("/comparison")
async def comparison_info():
    """멀티에이전트 vs 단일체인 비교 정보"""
    return {
        "architecture": "Single LangChain",
        "processing_type": "Sequential",
        "llm_instances": 1,
        "expected_performance": {
            "response_time": "25-30초 (예상)",
            "search_approach": "순차적 카테고리 검색",
            "parallelization": "없음",
            "optimization": "기본 LangChain 체인"
        },
        "vs_multi_agent": {
            "multi_agent_time": "~7초",
            "single_chain_time": "~25-30초",
            "performance_difference": "약 75% 느림 (예상)",
            "advantages": ["단순한 구조", "적은 메모리 사용"],
            "disadvantages": ["느린 응답", "순차 처리", "병렬 최적화 없음"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🔗 LangChain 단일 체인 챗봇 시작!")
    print("📍 서버: http://localhost:8003")
    print("🔍 진단: http://localhost:8003/health")
    print("📊 비교 정보: http://localhost:8003/comparison")
    uvicorn.run(app, host="0.0.0.0", port=8003)
