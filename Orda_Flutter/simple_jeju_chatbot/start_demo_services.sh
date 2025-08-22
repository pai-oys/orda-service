#!/bin/bash

# 데모데이용 개인화된 제주도 여행 챗봇 서비스 시작 스크립트

echo "🌴 데모데이용 개인화된 제주도 여행 챗봇 서비스를 시작합니다..."

# 현재 디렉토리 확인
if [ ! -f "demo_personalized_chatbot.py" ]; then
    echo "❌ simple_jeju_chatbot 디렉토리에서 실행해주세요."
    exit 1
fi

# 필요한 파일들 확인
if [ ! -f "demo_user_data.json" ]; then
    echo "❌ demo_user_data.json 파일이 없습니다."
    exit 1
fi

# Python 가상환경 활성화 (선택사항)
if [ -d "venv" ]; then
    echo "📦 가상환경 활성화 중..."
    source venv/bin/activate
fi

# 필요한 패키지 설치 확인
echo "📦 필요한 패키지 확인 중..."
pip install -q fastapi uvicorn python-dotenv langchain-upstage langgraph httpx

# RAG 서비스 확인 (포트 8002)
echo "🔍 RAG 서비스 상태 확인 중..."
if ! curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "⚠️  RAG 서비스(포트 8002)가 실행되지 않았습니다."
    echo "   advanced_jeju_chatbot 서비스를 먼저 시작해주세요."
    echo "   cd ../advanced_jeju_chatbot && python -m uvicorn api.main:app --host 0.0.0.0 --port 8002"
else
    echo "✅ RAG 서비스 연결 확인됨"
fi

# 백그라운드에서 개인화된 챗봇 서버 시작
echo "🚀 개인화된 챗봇 서버 시작 중... (포트 8004)"
python demo_chatbot_server.py &
DEMO_PID=$!

# 서버 시작 대기
sleep 3

# 서버 상태 확인
if curl -s http://localhost:8004/ > /dev/null 2>&1; then
    echo "✅ 개인화된 챗봇 서버가 성공적으로 시작되었습니다!"
    echo ""
    echo "🎯 데모 정보:"
    echo "   - 개인화된 챗봇 서버: http://localhost:8004"
    echo "   - 등록된 사용자: 30명"
    echo "   - 성향: 에겐남, 에겐녀, 테토남, 테토녀"
    echo ""
    echo "📱 Flutter 앱에서 이름을 입력하여 개인화된 채팅을 시작하세요!"
    echo ""
    echo "⚠️  서버를 종료하려면 Ctrl+C를 누르세요."
    
    # 서버 로그 표시
    wait $DEMO_PID
else
    echo "❌ 개인화된 챗봇 서버 시작에 실패했습니다."
    kill $DEMO_PID 2>/dev/null
    exit 1
fi