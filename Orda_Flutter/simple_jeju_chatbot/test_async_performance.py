#!/usr/bin/env python3
"""
비동기 처리 성능 테스트 스크립트

이 스크립트는 RAG 서버와 멀티에이전트 시스템의 비동기 처리 성능을 테스트합니다.
"""

import asyncio
import httpx
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# 서버 URL 설정
RAG_SERVER_URL = "http://localhost:8002"
MULTIAGENT_URL = "http://localhost:8001"

class AsyncPerformanceTester:
    """비동기 처리 성능 테스터"""
    
    def __init__(self):
        self.test_queries = [
            "제주도 호텔 추천",
            "제주도 관광지 추천", 
            "제주도 맛집 추천",
            "제주도 이벤트 추천"
        ]
    
    async def test_rag_server_direct(self) -> Dict[str, Any]:
        """RAG 서버 직접 테스트"""
        print("🔍 RAG 서버 직접 테스트 시작...")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # 순차 처리 테스트
            sequential_start = time.time()
            sequential_results = []
            
            for i, query in enumerate(self.test_queries):
                start = time.time()
                
                response = await client.post(
                    f"{RAG_SERVER_URL}/search",
                    json={
                        "query": query,
                        "top_k": 3,
                        "search_type": "similarity",
                        "filters": {}
                    }
                )
                
                duration = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    sequential_results.append({
                        "query": query,
                        "duration": duration,
                        "results": len(data.get("sources", [])),
                        "processing_time": data.get("processing_time", 0)
                    })
                    print(f"  순차 {i+1}: {duration:.2f}초 ({len(data.get('sources', []))}개 결과)")
                else:
                    sequential_results.append({
                        "query": query,
                        "duration": duration,
                        "error": f"HTTP {response.status_code}"
                    })
            
            sequential_total = time.time() - sequential_start
            
            # 병렬 처리 테스트
            parallel_start = time.time()
            
            tasks = []
            for query in self.test_queries:
                task = asyncio.create_task(
                    client.post(
                        f"{RAG_SERVER_URL}/search",
                        json={
                            "query": query,
                            "top_k": 3,
                            "search_type": "similarity",
                            "filters": {}
                        }
                    )
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            parallel_total = time.time() - parallel_start
            
            # 병렬 결과 정리
            parallel_results = []
            for i, (query, response) in enumerate(zip(self.test_queries, responses)):
                if isinstance(response, Exception):
                    parallel_results.append({
                        "query": query,
                        "error": str(response)
                    })
                elif response.status_code == 200:
                    data = response.json()
                    parallel_results.append({
                        "query": query,
                        "results": len(data.get("sources", [])),
                        "processing_time": data.get("processing_time", 0)
                    })
                else:
                    parallel_results.append({
                        "query": query,
                        "error": f"HTTP {response.status_code}"
                    })
            
            speedup = ((sequential_total - parallel_total) / sequential_total * 100) if sequential_total > 0 else 0
            
            return {
                "test_type": "rag_server_direct",
                "sequential": {
                    "total_time": sequential_total,
                    "results": sequential_results
                },
                "parallel": {
                    "total_time": parallel_total,
                    "results": parallel_results
                },
                "performance": {
                    "speedup_percentage": speedup,
                    "time_saved": sequential_total - parallel_total,
                    "efficiency_ratio": sequential_total / parallel_total if parallel_total > 0 else 0
                }
            }
    
    async def test_multiagent_system(self) -> Dict[str, Any]:
        """멀티에이전트 시스템 테스트"""
        print("🤖 멀티에이전트 시스템 테스트 시작...")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(f"{MULTIAGENT_URL}/test/async")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "error": f"멀티에이전트 테스트 실패: HTTP {response.status_code}"
                    }
        except Exception as e:
            return {
                "error": f"멀티에이전트 테스트 오류: {str(e)}"
            }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 비동기 성능 테스트"""
        print("🚀 종합 비동기 성능 테스트 시작!")
        print("=" * 60)
        
        start_time = time.time()
        
        # 병렬로 두 테스트 실행
        rag_task = asyncio.create_task(self.test_rag_server_direct())
        multiagent_task = asyncio.create_task(self.test_multiagent_system())
        
        rag_result, multiagent_result = await asyncio.gather(
            rag_task, multiagent_task, return_exceptions=True
        )
        
        total_time = time.time() - start_time
        
        # 결과 정리
        result = {
            "comprehensive_test": {
                "total_test_time": total_time,
                "rag_server_direct": rag_result if not isinstance(rag_result, Exception) else {"error": str(rag_result)},
                "multiagent_system": multiagent_result if not isinstance(multiagent_result, Exception) else {"error": str(multiagent_result)},
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # 결과 출력
        self.print_results(result)
        
        return result
    
    def print_results(self, result: Dict[str, Any]):
        """테스트 결과 출력"""
        print("\n" + "=" * 60)
        print("📊 비동기 처리 성능 테스트 결과")
        print("=" * 60)
        
        comp_test = result["comprehensive_test"]
        
        # RAG 서버 직접 테스트 결과
        if "rag_server_direct" in comp_test and "error" not in comp_test["rag_server_direct"]:
            rag = comp_test["rag_server_direct"]
            print("\n🔍 RAG 서버 직접 테스트:")
            print(f"  순차 처리: {rag['sequential']['total_time']:.2f}초")
            print(f"  병렬 처리: {rag['parallel']['total_time']:.2f}초")
            print(f"  성능 향상: {rag['performance']['speedup_percentage']:.1f}%")
            print(f"  시간 절약: {rag['performance']['time_saved']:.2f}초")
        
        # 멀티에이전트 시스템 결과
        if "multiagent_system" in comp_test and "error" not in comp_test["multiagent_system"]:
            multi = comp_test["multiagent_system"]
            if "multiagent_test" in multi:
                ma_test = multi["multiagent_test"]
                print("\n🤖 멀티에이전트 시스템:")
                print(f"  순차 처리: {ma_test['sequential']['total_time']:.2f}초")
                print(f"  병렬 처리: {ma_test['parallel']['total_time']:.2f}초")
                print(f"  성능 향상: {ma_test['performance']['speedup_percentage']:.1f}%")
                print(f"  시간 절약: {ma_test['performance']['time_saved']:.2f}초")
                
                if "comparison" in multi:
                    comp = multi["comparison"]
                    print(f"  네트워크 오버헤드: {comp['overhead']:.2f}초")
        
        print(f"\n⏱️  전체 테스트 시간: {comp_test['total_test_time']:.2f}초")
        print("=" * 60)

async def main():
    """메인 함수"""
    tester = AsyncPerformanceTester()
    
    try:
        result = await tester.run_comprehensive_test()
        
        # 결과를 JSON 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"async_test_result_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 테스트 결과가 {filename}에 저장되었습니다.")
        
    except KeyboardInterrupt:
        print("\n❌ 테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    print("🧪 비동기 처리 성능 테스트 도구")
    print("RAG 서버(8002)와 멀티에이전트(8001)가 실행 중인지 확인해주세요.")
    print()
    
    asyncio.run(main())
