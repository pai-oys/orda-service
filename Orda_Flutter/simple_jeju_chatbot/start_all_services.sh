#!/bin/bash

# 제주도 챗봇 모든 서비스 자동 실행 스크립트
# 한번에 모든 서비스를 백그라운드로 실행합니다

echo "🌴 제주도 챗봇 서비스 자동 실행 시작!"
echo "=================================="

# 기존 프로세스 종료
echo "🧹 기존 서비스 정리 중..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "uv run python smart_chatbot.py" 2>/dev/null || true
pkill -f "uv run python single_chain_baseline.py" 2>/dev/null || true
pkill -f "streamlit run streamlit_comparison.py" 2>/dev/null || true

sleep 3

# 로그 디렉토리 생성
mkdir -p logs

echo "🚀 서비스 실행 중..."

# 1. RAG 서비스 실행 (포트 8002)
echo "📡 RAG 서비스 시작 (포트 8002)..."
cd /Users/ohyooseok/Orda_Flutter/advanced_jeju_chatbot
nohup /Users/ohyooseok/miniconda3/bin/python api/main.py > /Users/ohyooseok/Orda_Flutter/simple_jeju_chatbot/logs/rag_service.log 2>&1 &
RAG_PID=$!
echo "   ✅ RAG 서비스 PID: $RAG_PID"

# RAG 서비스가 준비될 때까지 대기
echo "   ⏳ RAG 서비스 준비 대기..."
for i in {1..30}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "   ✅ RAG 서비스 준비 완료!"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "   ❌ RAG 서비스 시작 실패"
        exit 1
    fi
done

# 2. 멀티에이전트 시스템 실행 (포트 8001)
echo "🤖 멀티에이전트 시스템 시작 (포트 8001)..."
cd /Users/ohyooseok/Orda_Flutter/simple_jeju_chatbot
export PATH="$HOME/.local/bin:$PATH"
nohup uv run python smart_chatbot.py > logs/multi_agent.log 2>&1 &
MULTI_PID=$!
echo "   ✅ 멀티에이전트 PID: $MULTI_PID"

# 3. 단일체인 시스템 실행 (포트 8003)
echo "🔗 단일체인 시스템 시작 (포트 8003)..."
nohup uv run python single_chain_baseline.py > logs/single_chain.log 2>&1 &
SINGLE_PID=$!
echo "   ✅ 단일체인 PID: $SINGLE_PID"

# 4. Streamlit 비교 앱 실행 (포트 8501)
echo "📊 Streamlit 비교 앱 시작 (포트 8501)..."
nohup uv run streamlit run streamlit_comparison.py --server.port 8501 > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "   ✅ Streamlit PID: $STREAMLIT_PID"

echo ""
echo "⏳ 모든 서비스 준비 대기 중..."
sleep 10

echo ""
echo "🎉 모든 서비스 실행 완료!"
echo "========================"
echo ""
echo "📋 서비스 정보:"
echo "- 🔍 RAG 서비스:      http://localhost:8002 (PID: $RAG_PID)"
echo "- 🤖 멀티에이전트:    http://localhost:8001 (PID: $MULTI_PID)"
echo "- 🔗 단일체인:        http://localhost:8003 (PID: $SINGLE_PID)"
echo "- 📊 Streamlit 앱:    http://localhost:8501 (PID: $STREAMLIT_PID)"
echo ""
echo "📝 로그 파일:"
echo "- RAG 서비스:      logs/rag_service.log"
echo "- 멀티에이전트:    logs/multi_agent.log"
echo "- 단일체인:        logs/single_chain.log"
echo "- Streamlit:       logs/streamlit.log"
echo ""
echo "🔍 서비스 상태 확인:"
echo "curl http://localhost:8002/health  # RAG 서비스"
echo "curl http://localhost:8001/health  # 멀티에이전트"
echo "curl http://localhost:8003/health  # 단일체인"
echo ""
echo "🛑 모든 서비스 종료하려면:"
echo "./stop_all_services.sh"
echo ""
echo "📊 Streamlit 앱을 열려면:"
echo "open http://localhost:8501"
