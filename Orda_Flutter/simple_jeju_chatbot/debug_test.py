"""
🔧 제주도 챗봇 디버깅 테스트 스크립트
각 컴포넌트별로 단계적 테스트 수행
"""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage

async def main():
    # .env 파일 로딩
    load_dotenv()
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

    print("🔧 제주도 챗봇 디버깅 시작!")
    print("=" * 50)

    # 1. 환경변수 확인
    print("1️⃣ 환경변수 확인")
    if UPSTAGE_API_KEY:
        print(f"✅ UPSTAGE_API_KEY: {UPSTAGE_API_KEY[:10]}...")
    else:
        print("❌ UPSTAGE_API_KEY가 설정되지 않았습니다!")

    print()

    # 2. LangChain Upstage 연결 테스트
    print("2️⃣ LangChain Upstage 연결 테스트")
    try:
        llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")
        print("✅ LangChain Upstage 인스턴스 생성 성공")
        llm_created = True
    except Exception as e:
        print(f"❌ LangChain Upstage 연결 실패: {e}")
        llm_created = False
        llm = None

    print()

    # 3. RAG 서버 연결 테스트
    print("3️⃣ RAG 서버 연결 테스트")
    async def test_rag_connection():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "http://localhost:8002/chat",
                    json={"query": "제주도 테스트"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ RAG 서버 연결 성공 - 상태코드: {response.status_code}")
                    print(f"📊 응답 타입: {type(result)}")
                    print(f"📊 응답 키들: {list(result.keys()) if isinstance(result, dict) else 'dict가 아님'}")
                    
                    if isinstance(result, dict) and 'sources' in result:
                        print(f"📊 검색된 소스 수: {len(result.get('sources', []))}")
                    
                    return True
                else:
                    print(f"❌ RAG 서버 응답 오류 - 상태코드: {response.status_code}")
                    return False
                    
        except httpx.ConnectError as e:
            print(f"❌ RAG 서버 연결 실패 (ConnectError): {e}")
            return False
        except httpx.TimeoutException as e:
            print(f"❌ RAG 서버 연결 타임아웃: {e}")
            return False
        except Exception as e:
            print(f"❌ RAG 서버 연결 오류: {e}")
            return False

    # RAG 연결 테스트 실행
    rag_connected = await test_rag_connection()
    print()

    # 4. LLM 호출 테스트
    print("4️⃣ LLM 호출 테스트")
    llm_working = False
    if llm_created and llm:
        try:
            response = await llm.ainvoke("안녕하세요!")
            print(f"✅ LLM 호출 성공")
            print(f"📊 응답 타입: {type(response)}")
            print(f"📊 응답 내용: {response.content[:100]}...")
            llm_working = True
        except Exception as e:
            print(f"❌ LLM 호출 실패: {e}")
    else:
        print("❌ LLM 인스턴스가 생성되지 않아 테스트 불가")

    print()

    # 5. 간단한 프로필 추출 테스트
    print("5️⃣ 프로필 추출 테스트")
    profile_working = False
    if llm_working and llm:
        try:
            prompt = """다음 사용자 메시지에서 여행 정보를 추출해주세요.

사용자 메시지: 여자친구랑 2박3일로 제주도 여행가려고해

다음 정보를 JSON 형태로 추출해주세요:
{
    "duration": "여행 기간",
    "group_type": "여행 유형",
    "interests": ["관심사 배열"]
}

JSON만 출력해주세요:"""
            
            response = await llm.ainvoke(prompt)
            print(f"✅ 프로필 추출 테스트 성공")
            print(f"📊 응답: {response.content}")
            
            # JSON 파싱 테스트
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            try:
                parsed = json.loads(content)
                print(f"✅ JSON 파싱 성공: {parsed}")
                profile_working = True
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 실패: {e}")
                print(f"원본 텍스트: {content}")
                
        except Exception as e:
            print(f"❌ 프로필 추출 테스트 실패: {e}")
    else:
        print("❌ LLM이 작동하지 않아 테스트 불가")

    print()

    # 6. 각 에이전트별 쿼리 생성 테스트
    print("6️⃣ 에이전트별 쿼리 생성 테스트")
    if llm_working and llm:
        profile_summary = "기간: 2박3일 | 여행 유형: 커플"
        
        agents = {
            "숙박": "사용자 프로필을 바탕으로 제주도 숙박 장소 검색을 위한 자연어 쿼리를 생성해주세요.\n자연어 검색 쿼리 한 문장으로 출력해주세요 (SQL이나 코드가 아닌 일반 검색어):\n예시: '제주도 커플 2박3일 호텔 펜션 추천'",
            "관광": "사용자 프로필을 바탕으로 제주도 관광지 검색을 위한 자연어 쿼리를 생성해주세요.\n자연어 검색 쿼리 한 문장으로 출력해주세요 (SQL이나 코드가 아닌 일반 검색어):\n예시: '제주도 커플 액티비티 관광지 포토스팟 추천'",
            "음식": "사용자 프로필을 바탕으로 제주도 맛집 검색을 위한 자연어 쿼리를 생성해주세요.\n자연어 검색 쿼리 한 문장으로 출력해주세요 (SQL이나 코드가 아닌 일반 검색어):\n예시: '제주도 커플 데이트 맛집 흑돼지 해산물 추천'",
            "행사": "사용자 프로필을 바탕으로 제주도 행사/이벤트 검색을 위한 자연어 쿼리를 생성해주세요.\n자연어 검색 쿼리 한 문장으로 출력해주세요 (SQL이나 코드가 아닌 일반 검색어):\n예시: '제주도 커플 축제 이벤트 행사 체험 프로그램'"
        }
        
        for agent_name, prompt_template in agents.items():
            try:
                prompt = f"{prompt_template}\n\n사용자 프로필: {profile_summary}\n\n검색 쿼리만 출력해줘:"
                response = await llm.ainvoke(prompt)
                query = response.content.strip()
                print(f"✅ {agent_name} 에이전트: {query}")
                
                # 실제 RAG 검색 테스트
                if rag_connected:
                    try:
                        print(f"  └─ 검색 쿼리: '{query}'")
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            rag_response = await client.post(
                                "http://localhost:8002/chat",
                                json={"query": query}
                            )
                            
                            print(f"  └─ HTTP 상태코드: {rag_response.status_code}")
                            
                            if rag_response.status_code == 200:
                                rag_result = rag_response.json()
                                sources_count = len(rag_result.get('sources', []))
                                print(f"  └─ RAG 검색 결과: {sources_count}개")
                                
                                # 응답 구조 확인
                                print(f"  └─ 응답 키들: {list(rag_result.keys())}")
                                
                            else:
                                print(f"  └─ RAG 검색 실패: {rag_response.status_code}")
                                try:
                                    error_content = rag_response.text
                                    print(f"  └─ 오류 내용: {error_content[:200]}...")
                                except:
                                    print("  └─ 오류 내용을 읽을 수 없음")
                                
                    except httpx.ConnectError as e:
                        print(f"  └─ RAG 연결 오류: {e}")
                    except httpx.TimeoutException as e:
                        print(f"  └─ RAG 타임아웃: {e}")
                    except httpx.RequestError as e:
                        print(f"  └─ RAG 요청 오류: {e}")
                    except Exception as e:
                        print(f"  └─ RAG 검색 오류 ({type(e).__name__}): {e}")
                
            except Exception as e:
                print(f"❌ {agent_name} 에이전트 실패: {e}")
    else:
        print("❌ LLM이 작동하지 않아 테스트 불가")

    print()

    # 7. LangGraph import 테스트
    print("7️⃣ LangGraph 라이브러리 테스트")
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        print("✅ LangGraph 라이브러리 import 성공")
        langgraph_ok = True
    except ImportError as e:
        print(f"❌ LangGraph 라이브러리 import 실패: {e}")
        print("해결방법: pip install langgraph")
        langgraph_ok = False

    print()

    # 종합 결과
    print("📋 종합 결과")
    print("=" * 50)
    print(f"환경변수: {'✅' if UPSTAGE_API_KEY else '❌'}")
    print(f"LangChain: {'✅' if llm_created else '❌'}")
    print(f"RAG 서버: {'✅' if rag_connected else '❌'}")
    print(f"LLM 호출: {'✅' if llm_working else '❌'}")
    print(f"프로필 추출: {'✅' if profile_working else '❌'}")
    print(f"LangGraph: {'✅' if langgraph_ok else '❌'}")

    if all([UPSTAGE_API_KEY, llm_created, rag_connected, llm_working, profile_working, langgraph_ok]):
        print("\n🎉 모든 테스트 통과! 시스템이 정상 작동할 것 같습니다.")
        print("만약 여전히 문제가 있다면 LangGraph 그래프 구성 부분을 확인해보세요.")
    else:
        print("\n⚠️ 일부 테스트 실패. 위의 실패 항목들을 먼저 해결해주세요.")
        
        if not UPSTAGE_API_KEY:
            print("💡 .env 파일에 UPSTAGE_API_KEY를 설정해주세요")
        if not rag_connected:
            print("💡 RAG 서버(포트 8002)가 실행되고 있는지 확인해주세요")
        if not langgraph_ok:
            print("💡 pip install langgraph 명령어로 라이브러리를 설치해주세요")

    print("\n🔧 디버깅 완료!")

if __name__ == "__main__":
    asyncio.run(main())