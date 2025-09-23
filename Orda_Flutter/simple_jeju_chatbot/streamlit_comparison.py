"""
⚔️ 멀티에이전트 vs 단일체인 성능 비교 Streamlit 앱
- 실시간 성능 비교
- 시각화된 결과 분석
- 논문용 실험 데이터 수집
"""

import streamlit as st
import asyncio
import httpx
import time
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime
from typing import Dict, List, Optional

def display_execution_logs(logs: List[Dict], title: str, container):
    """실행 로그를 Streamlit에 표시"""
    if not logs:
        container.info("실행 로그가 없습니다.")
        return
    
    with container:
        st.markdown(f"**📋 {title}**")
        
        # 로그 타입별 색상 지정
        type_colors = {
            "multi_agent": "🚀",
            "single_chain": "🔗", 
            "agent": "🤖",
            "sequential": "📝",
            "timing": "⏱️",
            "info": "ℹ️"
        }
        
        # 로그를 타임스탬프 순으로 정렬
        sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''))
        
        # 로그 표시 영역
        log_container = st.container()
        with log_container:
            for log in sorted_logs:
                timestamp = log.get('timestamp', '')
                log_type = log.get('type', 'info')
                message = log.get('message', '')
                
                # 이모지와 함께 로그 메시지 표시
                emoji = type_colors.get(log_type, "📄")
                st.text(f"{timestamp} {emoji} {message}")
        
        # 요약 통계
        total_logs = len(logs)
        st.caption(f"총 {total_logs}개의 로그 메시지")

# 페이지 설정
st.set_page_config(
    page_title="⚔️ 멀티에이전트 vs 단일체인 비교",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .system-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .multi-agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .single-chain-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .winner-banner {
        background: linear-gradient(90deg, #00d4aa, #00d4aa);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'test_results' not in st.session_state:
    st.session_state.test_results = []
if 'current_test' not in st.session_state:
    st.session_state.current_test = None

# 메인 헤더
st.markdown('<h1 class="main-header">⚔️ 멀티에이전트 vs 단일체인 성능 비교</h1>', unsafe_allow_html=True)
st.markdown("---")

# 사이드바 - 시스템 정보
with st.sidebar:
    st.header("🔧 시스템 정보")
    
    # 서버 상태 확인
    st.subheader("서버 상태")
    
    async def check_server_status(url: str, name: str):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    return f"✅ {name}: 정상"
                else:
                    return f"❌ {name}: 오류 ({response.status_code})"
        except Exception as e:
            return f"❌ {name}: 연결 실패"
    
    # 서버 상태 체크 버튼
    if st.button("🔍 서버 상태 확인"):
        with st.spinner("서버 상태 확인 중..."):
            try:
                # 동기 방식으로 변경
                import requests
                
                # 멀티에이전트 서버 체크
                try:
                    response = requests.get("http://localhost:8001/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ 멀티에이전트 (8001): 정상")
                    else:
                        st.error(f"❌ 멀티에이전트 (8001): 오류 ({response.status_code})")
                except Exception as e:
                    st.error(f"❌ 멀티에이전트 (8001): 연결 실패")
                
                # 단일체인 서버 체크
                try:
                    response = requests.get("http://localhost:8003/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ 단일체인 (8003): 정상")
                    else:
                        st.error(f"❌ 단일체인 (8003): 오류 ({response.status_code})")
                except Exception as e:
                    st.error(f"❌ 단일체인 (8003): 연결 실패")
                
                # RAG 서버 체크
                try:
                    response = requests.get("http://localhost:8002/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ RAG 서비스 (8002): 정상")
                    else:
                        st.error(f"❌ RAG 서비스 (8002): 오류 ({response.status_code})")
                except Exception as e:
                    st.error(f"❌ RAG 서비스 (8002): 연결 실패")
                    
            except Exception as e:
                st.error(f"상태 확인 실패: {e}")
    
    st.markdown("---")
    
    # 테스트 설정
    st.subheader("⚙️ 테스트 설정")
    test_iterations = st.slider("테스트 반복 횟수", 1, 5, 1)
    include_streaming = st.checkbox("스트리밍 테스트 포함", value=True)
    
    st.markdown("---")
    
    # 결과 통계
    if st.session_state.test_results:
        st.subheader("📊 누적 통계")
        df = pd.DataFrame(st.session_state.test_results)
        
        if not df.empty:
            avg_multi = df['multi_agent_time'].mean()
            avg_single = df['single_chain_time'].mean()
            speedup = ((avg_single - avg_multi) / avg_single * 100)
            
            st.metric("평균 속도 향상", f"{speedup:.1f}%")
            st.metric("평균 시간 단축", f"{avg_single - avg_multi:.1f}초")
            st.metric("총 테스트 수", len(df))

# 메인 컨텐츠 영역
col1, col2 = st.columns(2)

# 멀티에이전트 시스템 카드
with col1:
    st.markdown("""
    <div class="system-card multi-agent-card">
        <h3>🤖 멀티에이전트 시스템</h3>
        <p>• LangGraph 기반 병렬 처리</p>
        <p>• 6개 전문 에이전트</p>
        <p>• 실시간 스트리밍 지원</p>
        <p>• 예상 응답시간: ~7초</p>
    </div>
    """, unsafe_allow_html=True)

# 단일체인 시스템 카드
with col2:
    st.markdown("""
    <div class="system-card single-chain-card">
        <h3>🔗 단일체인 시스템</h3>
        <p>• LangChain 기반 순차 처리</p>
        <p>• 1개 범용 LLM</p>
        <p>• 전통적인 RAG 패턴</p>
        <p>• 예상 응답시간: ~25-30초</p>
    </div>
    """, unsafe_allow_html=True)

# 🎯 교수님을 위한 핵심 증명 요약
st.markdown("### 🎓 **연구 증명 요약** (교수님께)")
col1, col2 = st.columns(2)

with col1:
    st.success("🚀 **멀티에이전트 시스템 (LangGraph)**")
    st.write("**✅ 병렬 처리 구현:**")
    st.write("• `asyncio.gather()`로 4개 에이전트 **동시 실행**")
    st.write("• 각 에이전트가 **독립적으로** 쿼리 생성 + 검색")
    st.write("• 전체 시간 = **가장 오래 걸린 에이전트** 시간")
    st.write("• **비동기 병렬 처리**로 성능 최적화")

with col2:
    st.warning("🔗 **단일체인 시스템 (LangChain)**")
    st.write("**✅ 순차 처리 구현:**")
    st.write("• 1개 에이전트가 **카테고리별 순서대로** 처리")
    st.write("• 쿼리 생성 → 검색 → 다음 카테고리 **반복**")
    st.write("• 전체 시간 = **모든 단계 시간의 합계**")
    st.write("• **전통적인 순차 처리** 방식")

st.info("🎯 **핵심 증명 포인트**: 아래 테스트 결과에서 **시각적 차트**와 **상세 시간 분석**을 통해 병렬 vs 순차 처리의 차이를 명확히 확인할 수 있습니다.")

st.markdown("---")

# 테스트 입력 섹션
st.header("🧪 성능 비교 테스트")

# 테스트 메시지 입력
test_message = st.text_area(
    "테스트할 여행 질문을 입력하세요:",
    value="가족 4명이 3박4일 제주도 여행을 계획 중입니다. 바다를 좋아하고 맛집 탐방에 관심이 많습니다.",
    height=100
)

# 테스트 실행 버튼들
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🤖 멀티에이전트만 테스트", type="secondary"):
        if test_message.strip():
            with st.spinner("멀티에이전트 시스템 테스트 중..."):
                start_time = time.time()
                try:
                    response = requests.post(
                        "http://localhost:8001/chat",
                        json={
                            "content": test_message,
                            "session_id": f"streamlit_multi_{int(time.time())}",
                            "conversation_history": [],
                            "user_profile": {},
                            "profile_completion": 0.0
                        },
                        timeout=120
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        data = response.json()
                        processing_time = end_time - start_time
                        search_duration = data.get('search_duration', 0.0)
                        search_queries = data.get('search_queries', {})
                        timing_details = data.get('timing_details', {})
                        
                        st.success(f"✅ 멀티에이전트 완료: {processing_time:.1f}초")
                        st.info(f"🔍 순수 검색 시간: {search_duration:.2f}초 (병렬)")
                        
                        # 사용된 쿼리 표시 (좌우로 길게)
                        if search_queries:
                            st.subheader("🎯 사용된 검색 쿼리")
                            col1, col2 = st.columns(2)
                            categories = list(search_queries.items())
                            mid = len(categories) // 2
                            
                            with col1:
                                for category, query in categories[:mid]:
                                    st.text(f"• {category}: {query}")
                            with col2:
                                for category, query in categories[mid:]:
                                    st.text(f"• {category}: {query}")
                        
                        # 상세 시간 분석 표시 (좌우로 길게)
                        if timing_details:
                            st.subheader("📊 상세 시간 분석")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                query_generation = timing_details.get('query_generation_time', 0)
                                task_creation = timing_details.get('task_creation_time', 0) * 1000
                                task_creation_ns = timing_details.get('task_creation_time_ns', 0)
                                parallel_execution = timing_details.get('parallel_execution_time', 0)
                                st.text(f"🧠 쿼리 생성: {query_generation:.2f}s")
                                st.text(f"⚙️  태스크 생성: {task_creation:.3f}ms")
                                if task_creation_ns > 0:
                                    st.text(f"   📏 {task_creation_ns:,}ns")
                                st.text(f"⚡ 병렬 실행: {parallel_execution:.2f}s")
                            
                            with col2:
                                result_processing = timing_details.get('result_processing_time', 0) * 1000
                                result_processing_ns = timing_details.get('result_processing_time_ns', 0)
                                st.text(f"📊 결과 처리: {result_processing:.3f}ms")
                                if result_processing_ns > 0:
                                    st.text(f"   📏 {result_processing_ns:,}ns")
                            
                            # 🎯 병렬 처리 증명 섹션
                            st.subheader("🚀 **병렬 처리 증명** (Multi-Agent)")
                            st.info("**핵심**: 4개 에이전트가 **동시에 시작**하여 **독립적으로** 쿼리 생성 + 검색을 수행합니다.")
                            
                            # 멀티에이전트 카테고리별 상세 시간 표시
                            category_timings = timing_details.get('category_timings', {})
                            if category_timings:
                                # 병렬 처리 시각화
                                st.markdown("**📋 각 에이전트별 독립 실행 시간:**")
                                
                                # 시간 막대 그래프로 병렬 처리 시각화
                                import pandas as pd
                                import plotly.express as px
                                
                                # 데이터 준비
                                agents = []
                                query_times = []
                                search_times = []
                                total_times = []
                                
                                for category, timing in category_timings.items():
                                    agents.append(f"{category.upper()}")
                                    query_times.append(timing['query_generation_time'])
                                    search_times.append(timing['search_time'])
                                    total_times.append(timing['total_time'])
                                
                                # 병렬 처리 시각화 차트
                                df_parallel = pd.DataFrame({
                                    'Agent': agents,
                                    'Query Generation': query_times,
                                    'Search Execution': search_times
                                })
                                
                                fig_parallel = px.bar(df_parallel, 
                                                    x='Agent', 
                                                    y=['Query Generation', 'Search Execution'],
                                                    title="🚀 멀티에이전트 병렬 처리 (각 에이전트 독립 실행)",
                                                    color_discrete_map={
                                                        'Query Generation': '#FF6B6B',
                                                        'Search Execution': '#4ECDC4'
                                                    })
                                fig_parallel.update_layout(
                                    yaxis_title="시간 (초)",
                                    xaxis_title="에이전트",
                                    height=400
                                )
                                st.plotly_chart(fig_parallel, use_container_width=True)
                                
                                # 병렬 처리 핵심 지표
                                max_total_time = max(total_times)
                                sum_total_time = sum(total_times)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("⚡ 실제 소요시간", f"{max_total_time:.2f}초", 
                                            help="병렬 처리: 가장 오래 걸린 에이전트 기준")
                                with col2:
                                    st.metric("📊 순차 처리시", f"{sum_total_time:.2f}초", 
                                            help="만약 순차로 했다면 걸렸을 시간")
                                with col3:
                                    speedup = sum_total_time / max_total_time
                                    st.metric("🚀 병렬 효율성", f"{speedup:.1f}배", 
                                            help="병렬 처리로 얻은 속도 향상")
                                
                                # LLM 호출 횟수 표시
                                llm_calls = data.get('llm_calls_count', 0)
                                if llm_calls > 0:
                                    st.info(f"🧠 **LLM 호출 횟수**: {llm_calls}회 (프로필 추출 1회 + 쿼리 생성 4회 + 응답 생성 1회)")
                                    
                                    # 비용 예상 (GPT-4 기준)
                                    estimated_cost = llm_calls * 0.03  # 대략적인 예상 비용 (달러)
                                    st.caption(f"💰 예상 비용: ~${estimated_cost:.3f} (GPT-4 기준)")
                                
                                # 상세 시간 테이블
                                st.markdown("**📋 에이전트별 상세 시간:**")
                                for category, timing in category_timings.items():
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.text(f"🤖 {category.upper()}")
                                    with col2:
                                        st.text(f"🧠 쿼리: {timing['query_generation_time']:.2f}s")
                                    with col3:
                                        st.text(f"🔍 검색: {timing['search_time']:.2f}s")
                                    with col4:
                                        st.text(f"⏱️ 총: {timing['total_time']:.2f}s")
                                st.text(f"⏱️  전체 시간: {search_duration:.2f}s")
                        
                        # 실행 로그 표시
                        execution_logs = data.get('execution_logs', [])
                        if execution_logs:
                            st.subheader("📋 실행 로그 (병렬 처리 증명)")
                            log_container = st.expander("멀티에이전트 실행 로그 보기", expanded=False)
                            display_execution_logs(execution_logs, "멀티에이전트 실행 과정", log_container)
                        
                        st.text_area("응답 결과", data.get('response', ''), height=200)
                    else:
                        st.error(f"❌ 오류: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ 테스트 실패: {e}")
        else:
            st.warning("테스트 메시지를 입력해주세요.")

with col2:
    if st.button("🔗 단일체인만 테스트", type="secondary"):
        if test_message.strip():
            with st.spinner("단일체인 시스템 테스트 중..."):
                start_time = time.time()
                try:
                    response = requests.post(
                        "http://localhost:8003/chat",
                        json={
                            "content": test_message,
                            "session_id": f"streamlit_single_{int(time.time())}",
                            "conversation_history": [],
                            "user_profile": {},
                            "profile_completion": 0.0
                        },
                        timeout=120
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        data = response.json()
                        processing_time = end_time - start_time
                        search_duration = data.get('search_duration', 0.0)
                        search_queries = data.get('search_queries', {})
                        timing_details = data.get('timing_details', {})
                        
                        st.success(f"✅ 단일체인 완료: {processing_time:.1f}초")
                        st.info(f"🔍 순수 검색 시간: {search_duration:.2f}초 (순차)")
                        
                        # 사용된 쿼리 표시 (좌우로 길게)
                        if search_queries:
                            st.subheader("🎯 사용된 검색 쿼리")
                            col1, col2 = st.columns(2)
                            categories = list(search_queries.items())
                            mid = len(categories) // 2
                            
                            with col1:
                                for category, query in categories[:mid]:
                                    st.text(f"• {category}: {query}")
                            with col2:
                                for category, query in categories[mid:]:
                                    st.text(f"• {category}: {query}")
                        
                        # 상세 시간 분석 표시 (좌우로 길게)
                        if timing_details:
                            st.subheader("📊 상세 시간 분석")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                query_generation = timing_details.get('query_generation_time', 0)
                                sequential_execution = timing_details.get('sequential_execution_time', 0)
                                st.text(f"🧠 총 쿼리 생성: {query_generation:.2f}s")
                                st.text(f"🔗 순차 실행: {sequential_execution:.2f}s")
                            
                            with col2:
                                st.text(f"⏱️  전체 시간: {search_duration:.2f}s")
                            
                            # LLM 호출 횟수 표시
                            llm_calls = data.get('llm_calls_count', 0)
                            if llm_calls > 0:
                                st.info(f"🧠 **LLM 호출 횟수**: {llm_calls}회 (프로필 추출 1회 + 쿼리 생성 4회 + 응답 생성 1회)")
                                
                                # 비용 예상 (GPT-4 기준)
                                estimated_cost = llm_calls * 0.03  # 대략적인 예상 비용 (달러)
                                st.caption(f"💰 예상 비용: ~${estimated_cost:.3f} (GPT-4 기준)")
                            
                            # 🎯 순차 처리 증명 섹션
                            st.subheader("🔗 **순차 처리 증명** (Single-Chain)")
                            st.info("**핵심**: 1개 에이전트가 **순서대로** 각 카테고리별로 쿼리 생성 → 검색을 **반복** 수행합니다.")
                            
                            # 카테고리별 상세 시간 표시
                            category_timings = timing_details.get('category_timings', {})
                            if category_timings:
                                # 순차 처리 시각화
                                st.markdown("**📋 카테고리별 순차 실행 시간:**")
                                
                                # 시간 막대 그래프로 순차 처리 시각화
                                import pandas as pd
                                import plotly.express as px
                                
                                # 데이터 준비
                                categories = []
                                query_times = []
                                search_times = []
                                cumulative_times = []
                                current_cumulative = 0
                                
                                for category, timing in category_timings.items():
                                    categories.append(f"{category.upper()}")
                                    query_times.append(timing['query_generation_time'])
                                    search_times.append(timing['search_time'])
                                    current_cumulative += timing['total_time']
                                    cumulative_times.append(current_cumulative)
                                
                                # 순차 처리 시각화 차트
                                df_sequential = pd.DataFrame({
                                    'Category': categories,
                                    'Query Generation': query_times,
                                    'Search Execution': search_times
                                })
                                
                                fig_sequential = px.bar(df_sequential, 
                                                      x='Category', 
                                                      y=['Query Generation', 'Search Execution'],
                                                      title="🔗 단일체인 순차 처리 (카테고리별 순서대로 실행)",
                                                      color_discrete_map={
                                                          'Query Generation': '#FF9F43',
                                                          'Search Execution': '#10AC84'
                                                      })
                                fig_sequential.update_layout(
                                    yaxis_title="시간 (초)",
                                    xaxis_title="처리 순서 (카테고리)",
                                    height=400
                                )
                                st.plotly_chart(fig_sequential, use_container_width=True)
                                
                                # 순차 처리 핵심 지표
                                total_query_time = sum(query_times)
                                total_search_time = sum(search_times)
                                total_time = total_query_time + total_search_time
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("⏱️ 총 소요시간", f"{total_time:.2f}초", 
                                            help="순차 처리: 모든 단계의 시간 합계")
                                with col2:
                                    st.metric("🧠 총 쿼리시간", f"{total_query_time:.2f}초", 
                                            help="4개 카테고리 쿼리 생성 시간 합계")
                                with col3:
                                    st.metric("🔍 총 검색시간", f"{total_search_time:.2f}초", 
                                            help="4개 카테고리 검색 시간 합계")
                                
                                # 순차 처리 단계별 시간 테이블
                                st.markdown("**📋 순차 처리 단계별 시간:**")
                                step = 1
                                for category, timing in category_timings.items():
                                    col1, col2, col3, col4, col5 = st.columns(5)
                                    with col1:
                                        st.text(f"#{step} {category.upper()}")
                                    with col2:
                                        st.text(f"🧠 쿼리: {timing['query_generation_time']:.2f}s")
                                    with col3:
                                        st.text(f"🔍 검색: {timing['search_time']:.2f}s")
                                    with col4:
                                        st.text(f"⏱️ 소계: {timing['total_time']:.2f}s")
                                    with col5:
                                        st.text(f"📊 누적: {cumulative_times[step-1]:.2f}s")
                                    step += 1
                        
                        # 실행 로그 표시
                        execution_logs = data.get('execution_logs', [])
                        if execution_logs:
                            st.subheader("📋 실행 로그 (순차 처리 증명)")
                            log_container = st.expander("단일체인 실행 로그 보기", expanded=False)
                            display_execution_logs(execution_logs, "단일체인 실행 과정", log_container)
                        
                        st.text_area("응답 결과", data.get('response', ''), height=200)
                    else:
                        st.error(f"❌ 오류: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ 테스트 실패: {e}")
        else:
            st.warning("테스트 메시지를 입력해주세요.")

with col3:
    # 비동기 처리 성능 테스트
    if st.button("🧪 비동기 테스트", type="secondary"):
        with st.spinner("비동기 처리 성능을 테스트하고 있습니다..."):
            try:
                # 멀티에이전트 비동기 테스트
                response = requests.get("http://localhost:8001/test/async", timeout=120)
                
                if response.status_code == 200:
                    test_result = response.json()
                    
                    # 멀티에이전트 테스트 결과
                    if "multiagent_test" in test_result:
                        multiagent = test_result["multiagent_test"]
                        
                        st.success("✅ 비동기 테스트 완료!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 멀티에이전트 성능")
                            st.metric("순차 처리", f"{multiagent['sequential']['total_time']:.2f}초")
                            st.metric("병렬 처리", f"{multiagent['parallel']['total_time']:.2f}초")
                            st.metric("성능 향상", f"{multiagent['performance']['speedup_percentage']:.1f}%")
                        
                        with col2:
                            st.subheader("🔍 RAG 서버 성능")
                            if "rag_server_test" in test_result:
                                rag = test_result["rag_server_test"]
                                st.metric("순차 처리", f"{rag['sequential']['total_time']:.2f}초")
                                st.metric("병렬 처리", f"{rag['parallel']['total_time']:.2f}초")
                                st.metric("성능 향상", f"{rag['performance']['speedup_percentage']:.1f}%")
                        
                        # 비교 분석
                        if "comparison" in test_result:
                            comp = test_result["comparison"]
                            st.info(f"🌐 네트워크 오버헤드: {comp['overhead']:.2f}초")
                        
                        # 상세 결과 표시
                        with st.expander("📋 상세 테스트 결과"):
                            st.json(test_result)
                    
                    else:
                        st.error(f"테스트 실패: {test_result.get('error', '알 수 없는 오류')}")
                else:
                    st.error(f"테스트 요청 실패: HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"테스트 연결 실패: {e}")
            except Exception as e:
                st.error(f"테스트 중 오류 발생: {e}")
    
    if st.button("⚡ 동시 비교 테스트", type="primary"):
        if test_message.strip():
            # 프로그레스 바 생성
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 결과 표시용 컨테이너
            results_container = st.container()
            
            session_id = f"streamlit_comparison_{int(time.time())}"
            
            try:
                import threading
                import queue
                
                # 결과를 저장할 큐
                result_queue = queue.Queue()
                
                def test_multi_agent():
                    try:
                        start_time = time.time()
                        response = requests.post(
                            "http://localhost:8001/chat",
                            json={
                                "content": test_message,
                                "session_id": f"{session_id}_multi",
                                "conversation_history": [],
                                "user_profile": {},
                                "profile_completion": 0.0
                            },
                            timeout=120
                        )
                        end_time = time.time()
                        
                        if response.status_code == 200:
                            data = response.json()
                            result_queue.put({
                                'type': 'multi_agent',
                                'success': True,
                                'time': end_time - start_time,
                                'search_duration': data.get('search_duration', 0.0),
                                'response': data.get('response', ''),
                                'search_queries': data.get('search_queries', {}),
                                'timing_details': data.get('timing_details', {}),
                                'llm_calls_count': data.get('llm_calls_count', 0),
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            result_queue.put({
                                'type': 'multi_agent',
                                'success': False,
                                'error': f"HTTP {response.status_code}"
                            })
                    except Exception as e:
                        result_queue.put({
                            'type': 'multi_agent',
                            'success': False,
                            'error': str(e)
                        })
                
                def test_single_chain():
                    try:
                        start_time = time.time()
                        response = requests.post(
                            "http://localhost:8003/chat",
                            json={
                                "content": test_message,
                                "session_id": f"{session_id}_single",
                                "conversation_history": [],
                                "user_profile": {},
                                "profile_completion": 0.0
                            },
                            timeout=120
                        )
                        end_time = time.time()
                        
                        if response.status_code == 200:
                            data = response.json()
                            result_queue.put({
                                'type': 'single_chain',
                                'success': True,
                                'time': end_time - start_time,
                                'search_duration': data.get('search_duration', 0.0),
                                'response': data.get('response', ''),
                                'search_queries': data.get('search_queries', {}),
                                'timing_details': data.get('timing_details', {}),
                                'llm_calls_count': data.get('llm_calls_count', 0),
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            result_queue.put({
                                'type': 'single_chain',
                                'success': False,
                                'error': f"HTTP {response.status_code}"
                            })
                    except Exception as e:
                        result_queue.put({
                            'type': 'single_chain',
                            'success': False,
                            'error': str(e)
                        })
                
                # 스레드 시작
                status_text.text("🚀 두 시스템을 동시에 테스트하고 있습니다...")
                progress_bar.progress(10)
                
                multi_thread = threading.Thread(target=test_multi_agent)
                single_thread = threading.Thread(target=test_single_chain)
                
                multi_thread.start()
                single_thread.start()
                
                # 결과 수집
                results = {}
                completed = 0
                
                while completed < 2:
                    try:
                        result = result_queue.get(timeout=1)
                        results[result['type']] = result
                        completed += 1
                        progress_bar.progress(50 + (completed * 25))
                        
                        if result['success']:
                            status_text.text(f"✅ {result['type']} 완료: {result['time']:.1f}초")
                        else:
                            status_text.text(f"❌ {result['type']} 실패: {result['error']}")
                            
                    except queue.Empty:
                        continue
                
                # 스레드 종료 대기
                multi_thread.join()
                single_thread.join()
                
                progress_bar.progress(100)
                status_text.text("✅ 모든 테스트 완료!")
                
                # 결과 분석 및 표시
                if results.get('multi_agent', {}).get('success') and results.get('single_chain', {}).get('success'):
                    multi_time = results['multi_agent']['time']
                    single_time = results['single_chain']['time']
                    multi_search_time = results['multi_agent']['search_duration']
                    single_search_time = results['single_chain']['search_duration']
                    
                    speedup = ((single_time - multi_time) / single_time * 100)
                    time_saved = single_time - multi_time
                    search_speedup = ((single_search_time - multi_search_time) / single_search_time * 100) if single_search_time > 0 else 0
                    search_time_saved = single_search_time - multi_search_time
                    
                    # 승자 배너 (중앙 정렬)
                    # 빈 공간과 중앙 컬럼으로 레이아웃 조정
                    _, center_col, _ = st.columns([0.15, 0.7, 0.15])
                    
                    with center_col:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 25px;
                            border-radius: 15px;
                            text-align: center;
                            font-size: 20px;
                            font-weight: bold;
                            margin: 20px 0;
                            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                            width: 100%;
                        ">
                            🏆 멀티에이전트가 {speedup:.1f}% 더 빠릅니다!<br>
                            <span style="font-size: 16px;">({time_saved:.1f}초 단축)</span><br><br>
                            🔍 검색 시간: {search_speedup:.1f}% 단축<br>
                            <span style="font-size: 16px;">({search_time_saved:.1f}초)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 결과 저장
                    test_result = {
                        'timestamp': datetime.now().isoformat(),
                        'test_message': test_message,
                        'multi_agent_time': multi_time,
                        'single_chain_time': single_time,
                        'multi_agent_search_time': multi_search_time,
                        'single_chain_search_time': single_search_time,
                        'speedup_percentage': speedup,
                        'search_speedup_percentage': search_speedup,
                        'time_saved': time_saved,
                        'search_time_saved': search_time_saved,
                        'multi_agent_response': results['multi_agent']['response'],
                        'single_chain_response': results['single_chain']['response']
                    }
                    st.session_state.test_results.append(test_result)
                    
                    # 성능 메트릭 표시 (중앙 정렬, 3x2 레이아웃)
                    st.markdown("---")
                    st.markdown("### 📊 **성능 비교 결과**")
                    
                    # 첫 번째 줄: 전체 처리 시간 비교
                    _, metric_col, _ = st.columns([0.1, 0.8, 0.1])
                    with metric_col:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🤖 멀티에이전트", f"{multi_time:.1f}초", f"-{time_saved:.1f}초")
                        with col2:
                            st.metric("🔗 단일체인", f"{single_time:.1f}초", f"+{time_saved:.1f}초")
                        with col3:
                            st.metric("⚡ 속도 향상", f"{speedup:.1f}%", f"{single_time/multi_time:.1f}배")
                    
                    # 두 번째 줄: 검색 시간 비교
                    with metric_col:
                        col4, col5, col6 = st.columns(3)
                        
                        with col4:
                            st.metric("🔍 멀티 검색", f"{multi_search_time:.2f}초", f"-{search_time_saved:.2f}초")
                        with col5:
                            st.metric("🔍 단일 검색", f"{single_search_time:.2f}초", f"+{search_time_saved:.2f}초")
                        with col6:
                            st.metric("🚀 검색 향상", f"{search_speedup:.1f}%", f"{single_search_time/multi_search_time:.1f}배" if multi_search_time > 0 else "N/A")
                    
                    # 세 번째 줄: LLM 호출 횟수 비교
                    with metric_col:
                        col7, col8, col9 = st.columns(3)
                        
                        multi_llm_calls = results['multi_agent'].get('llm_calls_count', 0)
                        single_llm_calls = results['single_chain'].get('llm_calls_count', 0)
                        llm_difference = single_llm_calls - multi_llm_calls
                        
                        with col7:
                            st.metric("🧠 멀티 LLM", f"{multi_llm_calls}회", f"-{llm_difference}회" if llm_difference > 0 else f"+{abs(llm_difference)}회")
                        with col8:
                            st.metric("🧠 단일 LLM", f"{single_llm_calls}회", f"+{llm_difference}회" if llm_difference > 0 else f"-{abs(llm_difference)}회")
                        with col9:
                            if multi_llm_calls > 0 and single_llm_calls > 0:
                                if multi_llm_calls == single_llm_calls:
                                    st.metric("🤝 LLM 비용", "동일", "0회 차이")
                                else:
                                    cost_diff = abs(llm_difference) / max(multi_llm_calls, single_llm_calls) * 100
                                    st.metric("💰 비용 차이", f"{cost_diff:.1f}%", f"{abs(llm_difference)}회")
                            else:
                                st.metric("💰 비용 차이", "N/A", "정보 없음")
                    
                    # 상세 정보 표시 (중앙 정렬)
                    st.markdown("---")
                    
                    # 사용된 쿼리 정보 표시
                    multi_queries = results['multi_agent'].get('search_queries', {})
                    single_queries = results['single_chain'].get('search_queries', {})
                    multi_timing = results['multi_agent'].get('timing_details', {})
                    single_timing = results['single_chain'].get('timing_details', {})
                    
                    if multi_queries or single_queries:
                        st.subheader("🎯 사용된 검색 쿼리")
                        _, query_col, _ = st.columns([0.1, 0.8, 0.1])
                        
                        with query_col:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🤖 멀티에이전트 쿼리:**")
                                if multi_queries:
                                    for category, query in multi_queries.items():
                                        st.text(f"• {category}: {query}")
                                else:
                                    st.text("쿼리 정보 없음")
                            
                            with col2:
                                st.markdown("**🔗 단일체인 쿼리:**")
                                if single_queries:
                                    for category, query in single_queries.items():
                                        st.text(f"• {category}: {query}")
                                else:
                                    st.text("쿼리 정보 없음")
                    
                    # 🎯 병렬 vs 순차 처리 증명 섹션
                    st.subheader("🎯 **병렬 vs 순차 처리 증명**")
                    
                    # 핵심 차이점 설명
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("🚀 **멀티에이전트 (병렬 처리)**")
                        st.write("• 4개 에이전트가 **동시에 시작**")
                        st.write("• 각 에이전트가 **독립적으로** 실행")
                        st.write("• 전체 시간 = **가장 오래 걸린 에이전트** 시간")
                    
                    with col2:
                        st.warning("🔗 **단일체인 (순차 처리)**")
                        st.write("• 1개 에이전트가 **순서대로** 실행")
                        st.write("• 각 카테고리를 **차례대로** 처리")
                        st.write("• 전체 시간 = **모든 단계 시간의 합계**")
                    
                    # 상세 시간 분석 표시
                    if multi_timing or single_timing:
                        st.subheader("📊 상세 시간 분석")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🤖 멀티에이전트 시간 분석:**")
                            if multi_timing:
                                query_generation = multi_timing.get('query_generation_time', 0)
                                task_creation = multi_timing.get('task_creation_time', 0) * 1000
                                task_creation_ns = multi_timing.get('task_creation_time_ns', 0)
                                parallel_execution = multi_timing.get('parallel_execution_time', 0)
                                result_processing = multi_timing.get('result_processing_time', 0) * 1000
                                result_processing_ns = multi_timing.get('result_processing_time_ns', 0)
                                
                                st.text(f"🧠 쿼리 생성: {query_generation:.2f}s")
                                st.text(f"⚙️  태스크 생성: {task_creation:.3f}ms")
                                if task_creation_ns > 0:
                                    st.text(f"   📏 {task_creation_ns:,}ns")
                                st.text(f"⚡ 병렬 실행: {parallel_execution:.2f}s")
                                st.text(f"📊 결과 처리: {result_processing:.3f}ms")
                                if result_processing_ns > 0:
                                    st.text(f"   📏 {result_processing_ns:,}ns")
                                
                                # 멀티에이전트 카테고리별 상세 시간 표시
                                category_timings = multi_timing.get('category_timings', {})
                                if category_timings:
                                    st.text("📋 카테고리별 상세:")
                                    for category, timing in category_timings.items():
                                        st.text(f"  {category}: 쿼리({timing['query_generation_time']:.2f}s) + 검색({timing['search_time']:.2f}s)")
                            else:
                                st.text("시간 분석 정보 없음")
                        
                        with col2:
                            st.markdown("**🔗 단일체인 시간 분석:**")
                            if single_timing:
                                query_generation = single_timing.get('query_generation_time', 0)
                                sequential_execution = single_timing.get('sequential_execution_time', 0)
                                st.text(f"🧠 총 쿼리 생성: {query_generation:.2f}s")
                                st.text(f"🔗 순차 실행: {sequential_execution:.2f}s")
                                
                                # 카테고리별 상세 시간 표시
                                category_timings = single_timing.get('category_timings', {})
                                if category_timings:
                                    st.text("📋 카테고리별 상세:")
                                    for category, timing in category_timings.items():
                                        st.text(f"  {category}: 쿼리({timing['query_generation_time']:.2f}s) + 검색({timing['search_time']:.2f}s)")
                            else:
                                st.text("시간 분석 정보 없음")
                    
                    # 🎯 시각적 비교 차트
                    if multi_timing and single_timing:
                        multi_category_timings = multi_timing.get('category_timings', {})
                        single_category_timings = single_timing.get('category_timings', {})
                        
                        if multi_category_timings and single_category_timings:
                            st.subheader("📊 **병렬 vs 순차 처리 시각적 비교**")
                            
                            # 비교 차트 데이터 준비
                            import pandas as pd
                            import plotly.graph_objects as go
                            from plotly.subplots import make_subplots
                            
                            categories = list(multi_category_timings.keys())
                            
                            # 멀티에이전트 데이터
                            multi_query_times = [multi_category_timings[cat]['query_generation_time'] for cat in categories]
                            multi_search_times = [multi_category_timings[cat]['search_time'] for cat in categories]
                            
                            # 단일체인 데이터
                            single_query_times = [single_category_timings[cat]['query_generation_time'] for cat in categories]
                            single_search_times = [single_category_timings[cat]['search_time'] for cat in categories]
                            
                            # 서브플롯 생성
                            fig = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=('🚀 멀티에이전트 (병렬)', '🔗 단일체인 (순차)'),
                                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                            )
                            
                            # 멀티에이전트 차트
                            fig.add_trace(
                                go.Bar(name='쿼리 생성', x=[cat.upper() for cat in categories], y=multi_query_times, 
                                      marker_color='#FF6B6B', showlegend=True),
                                row=1, col=1
                            )
                            fig.add_trace(
                                go.Bar(name='검색 실행', x=[cat.upper() for cat in categories], y=multi_search_times, 
                                      marker_color='#4ECDC4', showlegend=True),
                                row=1, col=1
                            )
                            
                            # 단일체인 차트
                            fig.add_trace(
                                go.Bar(name='쿼리 생성', x=[cat.upper() for cat in categories], y=single_query_times, 
                                      marker_color='#FF9F43', showlegend=False),
                                row=1, col=2
                            )
                            fig.add_trace(
                                go.Bar(name='검색 실행', x=[cat.upper() for cat in categories], y=single_search_times, 
                                      marker_color='#10AC84', showlegend=False),
                                row=1, col=2
                            )
                            
                            fig.update_layout(
                                title="병렬 처리 vs 순차 처리 비교",
                                height=500,
                                barmode='stack'
                            )
                            fig.update_xaxes(title_text="에이전트/카테고리", row=1, col=1)
                            fig.update_xaxes(title_text="처리 순서", row=1, col=2)
                            fig.update_yaxes(title_text="시간 (초)", row=1, col=1)
                            fig.update_yaxes(title_text="시간 (초)", row=1, col=2)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 핵심 증명 지표
                            st.markdown("### 🎯 **핵심 증명 지표**")
                            
                            # 멀티에이전트 총 시간 (병렬 - 최대값)
                            multi_total_times = [multi_category_timings[cat]['total_time'] for cat in categories]
                            multi_actual_time = max(multi_total_times)
                            multi_if_sequential = sum(multi_total_times)
                            
                            # 단일체인 총 시간 (순차 - 합계)
                            single_total_times = [single_category_timings[cat]['total_time'] for cat in categories]
                            single_actual_time = sum(single_total_times)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🚀 멀티에이전트", f"{multi_actual_time:.2f}초", 
                                        help="병렬 처리: 가장 오래 걸린 에이전트")
                            with col2:
                                st.metric("🔗 단일체인", f"{single_actual_time:.2f}초", 
                                        help="순차 처리: 모든 단계 시간 합계")
                            with col3:
                                speedup = single_actual_time / multi_actual_time
                                st.metric("⚡ 속도 향상", f"{speedup:.1f}배", 
                                        help="병렬 처리로 얻은 성능 향상")
                            with col4:
                                efficiency = (1 - multi_actual_time / multi_if_sequential) * 100
                                st.metric("🎯 병렬 효율성", f"{efficiency:.1f}%", 
                                        help="병렬 처리 효율성")
                    
                    # 실행 로그 비교 (새로 추가)
                    multi_logs = results.get('multi_agent', {}).get('execution_logs', [])
                    single_logs = results.get('single_chain', {}).get('execution_logs', [])
                    
                    if multi_logs or single_logs:
                        st.subheader("📋 실행 로그 비교 (교수님께 증명)")
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            if multi_logs:
                                log_container = st.expander("🚀 멀티에이전트 병렬 실행 로그", expanded=False)
                                display_execution_logs(multi_logs, "병렬 처리 과정", log_container)
                            else:
                                st.info("멀티에이전트 로그 없음")
                        
                        with col2:
                            if single_logs:
                                log_container = st.expander("🔗 단일체인 순차 실행 로그", expanded=False)
                                display_execution_logs(single_logs, "순차 처리 과정", log_container)
                            else:
                                st.info("단일체인 로그 없음")
                    
                    # 응답 비교 (중앙 정렬)
                    st.subheader("📝 응답 비교")
                    _, response_col, _ = st.columns([0.05, 0.9, 0.05])
                    
                    with response_col:
                        col1, col2 = st.columns([1, 1])  # 동일한 비율로 좌우 분할
                        
                        with col1:
                            st.markdown("**🤖 멀티에이전트 응답:**")
                            st.text_area("멀티에이전트 응답", results['multi_agent']['response'], height=400, key="multi_response", label_visibility="collapsed")
                        
                        with col2:
                            st.markdown("**🔗 단일체인 응답:**")
                            st.text_area("단일체인 응답", results['single_chain']['response'], height=400, key="single_response", label_visibility="collapsed")
                
            except Exception as e:
                st.error(f"❌ 동시 테스트 실패: {e}")
                progress_bar.progress(0)
                status_text.text("")
        else:
            st.warning("테스트 메시지를 입력해주세요.")

# 결과 시각화 섹션
if st.session_state.test_results:
    st.markdown("---")
    st.header("📊 성능 분석 및 시각화")
    
    df = pd.DataFrame(st.session_state.test_results)
    
    # 성능 트렌드 차트
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df['multi_agent_search_time'],
        mode='lines+markers',
        name='멀티에이전트 (검색)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df['single_chain_search_time'],
        mode='lines+markers',
        name='단일체인 (검색)',
        line=dict(color='#f5576c', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="검색 시간 비교 트렌드",
        xaxis_title="테스트 번호",
        yaxis_title="응답 시간 (초)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # 검색 속도 향상 히스토그램
    fig2 = px.histogram(
        df, 
        x='search_speedup_percentage',
        nbins=10,
        title="검색 속도 향상 분포",
        labels={'search_speedup_percentage': '검색 속도 향상 (%)', 'count': '빈도'},
        color_discrete_sequence=['#00d4aa']
    )
    
    st.plotly_chart(fig2, width='stretch')
    
    # 통계 요약 (검색 시간 기준)
    st.subheader("📈 검색 시간 통계 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "평균 멀티에이전트 검색시간",
            f"{df['multi_agent_search_time'].mean():.2f}초",
            f"±{df['multi_agent_search_time'].std():.2f}초"
        )
    
    with col2:
        st.metric(
            "평균 단일체인 검색시간",
            f"{df['single_chain_search_time'].mean():.2f}초",
            f"±{df['single_chain_search_time'].std():.2f}초"
        )
    
    with col3:
        st.metric(
            "평균 검색 속도 향상",
            f"{df['search_speedup_percentage'].mean():.1f}%",
            f"±{df['search_speedup_percentage'].std():.1f}%"
        )
    
    with col4:
        st.metric(
            "평균 검색 시간 절약",
            f"{df['search_time_saved'].mean():.2f}초",
            f"±{df['search_time_saved'].std():.2f}초"
        )
    
    # 데이터 테이블
    st.subheader("📋 상세 테스트 결과")
    
    display_df = df[['timestamp', 'multi_agent_time', 'single_chain_time', 'speedup_percentage', 'time_saved']].copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
    display_df.columns = ['시간', '멀티에이전트(초)', '단일체인(초)', '속도향상(%)', '시간절약(초)']
    
    st.dataframe(display_df, use_container_width=True)
    
    # 결과 다운로드
    if st.button("📥 결과 데이터 다운로드 (CSV)"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="CSV 파일 다운로드",
            data=csv,
            file_name=f"chatbot_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# 결과 초기화 버튼
if st.session_state.test_results:
    if st.button("🗑️ 모든 결과 초기화", type="secondary"):
        st.session_state.test_results = []
        st.success("✅ 모든 테스트 결과가 초기화되었습니다.")
        st.rerun()

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌴 제주도 여행 챗봇 성능 비교 도구</p>
    <p>멀티에이전트 vs 단일체인 아키텍처 비교 연구용</p>
</div>
""", unsafe_allow_html=True)
