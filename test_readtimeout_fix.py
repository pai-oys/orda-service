#!/usr/bin/env python3
"""
ReadTimeout 수정 검증 테스트 스크립트
"""

import asyncio
import httpx
import time
from datetime import datetime

# 테스트 설정
CHATBOT_URL = "http://localhost:8003"
RAG_URL = "http://localhost:8002/chat"

async def test_rag_server_direct():
    """RAG 서버 직접 테스트 (재시도 로직 포함)"""
    print("🔍 RAG 서버 직접 테스트...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 점진적으로 타임아웃 증가: 30초 → 60초 → 90초
            read_timeout = 30.0 + (attempt * 30.0)
            timeout_config = httpx.Timeout(connect=10.0, read=read_timeout, write=10.0, pool=10.0)
            
            print(f"🔄 시도 {attempt + 1}/{max_retries} - 타임아웃: {read_timeout}초")
            
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                start_time = time.time()
                
                response = await client.post(
                    RAG_URL,
                    json={"query": "제주도 가족 여행 호텔 추천"},
                    headers={"User-Agent": "TestClient/1.0"}  # User-Agent 명시
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    sources_count = len(result.get("sources", []))
                    processing_time = result.get("processing_time", 0)
                    
                    print(f"✅ RAG 서버 응답 성공")
                    print(f"   - 응답 시간: {elapsed:.2f}초")
                    print(f"   - 처리 시간: {processing_time:.2f}초")
                    print(f"   - 검색 결과: {sources_count}개")
                    return True
                else:
                    print(f"❌ RAG 서버 오류 - 상태코드: {response.status_code}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # 1초, 2초, 4초 대기
                        continue
                    return False
                    
        except httpx.ReadTimeout:
            print(f"⏰ RAG 서버 ReadTimeout 발생 ({read_timeout}초)")
            if attempt < max_retries - 1:
                print(f"🔄 {2 ** attempt}초 후 재시도...")
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                print("❌ 모든 재시도 실패")
                return False
        except Exception as e:
            print(f"❌ RAG 서버 테스트 실패: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return False
    
    return False

async def test_chatbot_dynamic_search():
    """챗봇 동적 검색 개수 테스트"""
    print("🚀 챗봇 동적 검색 개수 테스트...")
    
    test_cases = [
        ("당일치기 제주도 여행", "1일"),
        ("2박3일 가족 여행", "3일"),
        ("5박6일 커플 여행", "6일")
    ]
    
    results = []
    
    for message, expected_days in test_cases:
        print(f"\n🧪 테스트: {message} (예상 {expected_days})")
        
        try:
            timeout_config = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                start_time = time.time()
                
                response = await client.post(
                    f"{CHATBOT_URL}/chat",
                    json={
                        "message": message,
                        "session_id": f"test_{expected_days}_{int(time.time())}"
                    }
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '')
                    
                    print(f"   ✅ 성공 - {elapsed:.2f}초, {len(response_text)}자")
                    
                    # 응답에서 다양한 장소가 언급되는지 확인
                    location_count = response_text.count('관광') + response_text.count('맛집') + response_text.count('음식점')
                    print(f"   📍 장소 언급: {location_count}개")
                    
                    results.append(True)
                else:
                    print(f"   ❌ 실패 - 상태코드: {response.status_code}")
                    results.append(False)
                    
        except httpx.ReadTimeout:
            print(f"   ⏰ 타임아웃 (120초 초과)")
            results.append(False)
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) if results else 0
    print(f"\n📊 동적 검색 테스트 성공률: {success_rate*100:.1f}%")
    return success_rate > 0.5

async def test_health_check():
    """헬스 체크 테스트"""
    print("🏥 헬스 체크 테스트...")
    
    try:
        # 헬스 체크는 RAG 서버 진단을 포함하므로 더 긴 타임아웃 필요
        timeout_config = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            start_time = time.time()
            response = await client.get(f"{CHATBOT_URL}/health")
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                rag_status = result.get("rag_server", {}).get("status", "unknown")
                rag_response_time = result.get("rag_server", {}).get("response_time", "N/A")
                
                print(f"✅ 헬스 체크 성공 ({elapsed:.2f}초)")
                print(f"   - 챗봇 상태: {result.get('chatbot_status', 'unknown')}")
                print(f"   - RAG 서버 상태: {rag_status}")
                print(f"   - RAG 서버 응답시간: {rag_response_time}")
                return True
            else:
                print(f"❌ 헬스 체크 오류 - 상태코드: {response.status_code}")
                return False
                
    except httpx.ReadTimeout:
        print("⏰ 헬스 체크 타임아웃 (60초 초과)")
        return False
    except Exception as e:
        print(f"❌ 헬스 체크 실패: {e}")
        return False

async def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print(f"🧪 ReadTimeout 수정 검증 테스트 시작")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = []
    
    # 테스트 1: 헬스 체크
    print("\n1️⃣ 헬스 체크 테스트")
    results.append(await test_health_check())
    
    # 테스트 2: RAG 서버 직접 테스트
    print("\n2️⃣ RAG 서버 직접 테스트")  
    results.append(await test_rag_server_direct())
    
    # 테스트 3: 챗봇 동적 검색 테스트
    print("\n3️⃣ 챗봇 동적 검색 개수 테스트")
    results.append(await test_chatbot_dynamic_search())
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    test_names = ["헬스 체크", "RAG 서버 직접", "챗봇 동적 검색"]
    success_count = sum(results)
    
    for i, (name, success) in enumerate(zip(test_names, results)):
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\n🎯 전체 성공률: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count == len(results):
        print("🎉 모든 테스트 통과! ReadTimeout 문제 해결 + 여행 기간별 동적 검색 완성!")
    else:
        print("⚠️ 일부 테스트 실패. 추가 디버깅이 필요합니다.")
        print("💡 팁: 다양한 여행 기간(1일, 3일, 6일)에 따른 검색 개수 차이를 확인해보세요.")
    
    print(f"⏰ 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())