#!/bin/bash

# 제주도 챗봇 모든 서비스 종료 스크립트

echo "🛑 제주도 챗봇 서비스 종료 중..."
echo "================================"

# 각 서비스 종료
echo "🧹 서비스 정리 중..."

# RAG 서비스 종료
echo "📡 RAG 서비스 종료..."
pkill -f "python.*main.py" 2>/dev/null && echo "   ✅ RAG 서비스 종료됨" || echo "   ℹ️  RAG 서비스 없음"

# 멀티에이전트 시스템 종료
echo "🤖 멀티에이전트 시스템 종료..."
pkill -f "uv run python smart_chatbot.py" 2>/dev/null && echo "   ✅ 멀티에이전트 종료됨" || echo "   ℹ️  멀티에이전트 없음"

# 단일체인 시스템 종료
echo "🔗 단일체인 시스템 종료..."
pkill -f "uv run python single_chain_baseline.py" 2>/dev/null && echo "   ✅ 단일체인 종료됨" || echo "   ℹ️  단일체인 없음"

# Streamlit 앱 종료
echo "📊 Streamlit 앱 종료..."
pkill -f "streamlit run streamlit_comparison.py" 2>/dev/null && echo "   ✅ Streamlit 종료됨" || echo "   ℹ️  Streamlit 없음"

# 포트 확인
echo ""
echo "🔍 포트 상태 확인:"
for port in 8001 8002 8003 8501; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "   ⚠️  포트 $port 아직 사용 중"
    else
        echo "   ✅ 포트 $port 해제됨"
    fi
done

echo ""
echo "🎉 모든 서비스 종료 완료!"
echo ""
echo "🚀 다시 시작하려면:"
echo "./start_all_services.sh"
