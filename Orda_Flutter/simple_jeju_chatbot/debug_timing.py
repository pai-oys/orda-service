#!/usr/bin/env python3
"""
멀티에이전트 시간 디버깅 스크립트
"""

import requests
import time

def test_multiagent_timing():
    """멀티에이전트 시간 디버깅"""
    print("🔍 멀티에이전트 시간 디버깅 시작...")
    
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:8001/chat",
        json={
            "content": "제주도 2박3일 여행 추천해주세요",
            "session_id": "debug_timing_test"
        },
        timeout=120
    )
    
    total_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        search_duration = data.get('search_duration', 0.0)
        
        print(f"\n📊 멀티에이전트 시간 분석:")
        print(f"   🔍 순수 검색 시간: {search_duration:.2f}초")
        print(f"   ⏱️  전체 처리 시간: {total_time:.2f}초")
        print(f"   🔄 기타 처리 시간: {total_time - search_duration:.2f}초")
        print(f"   📈 검색 비율: {search_duration/total_time*100:.1f}%")
        
    else:
        print(f"❌ 요청 실패: HTTP {response.status_code}")

if __name__ == "__main__":
    test_multiagent_timing()
