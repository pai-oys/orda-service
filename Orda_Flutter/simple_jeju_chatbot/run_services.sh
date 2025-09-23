#!/bin/bash

# 제주도 챗봇 서비스 실행 스크립트
# 각 터미널 탭에서 개별적으로 실행하세요

echo "🌴 제주도 챗봇 서비스 실행 가이드"
echo "=================================="
echo ""

echo "📋 실행 순서:"
echo "1. RAG 서비스 (포트 8002)"
echo "2. 멀티에이전트 시스템 (포트 8001)"
echo "3. 단일체인 시스템 (포트 8003)"
echo "4. Streamlit 비교 앱 (포트 8501)"
echo ""

echo "🔧 각 터미널 탭에서 실행할 명령어:"
echo ""

echo "=== 터미널 1: RAG 서비스 ==="
echo "cd /Users/ohyooseok/Orda_Flutter/advanced_jeju_chatbot"
echo "/Users/ohyooseok/miniconda3/bin/python api/main.py"
echo ""

echo "=== 터미널 2: 멀티에이전트 시스템 ==="
echo "cd /Users/ohyooseok/Orda_Flutter/simple_jeju_chatbot"
echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "uv run python smart_chatbot.py"
echo ""

echo "=== 터미널 3: 단일체인 시스템 ==="
echo "cd /Users/ohyooseok/Orda_Flutter/simple_jeju_chatbot"
echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "uv run python single_chain_baseline.py"
echo ""

echo "=== 터미널 4: Streamlit 비교 앱 ==="
echo "cd /Users/ohyooseok/Orda_Flutter/simple_jeju_chatbot"
echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "uv run streamlit run streamlit_comparison.py --server.port 8501"
echo ""

echo "✅ 모든 서비스가 실행되면:"
echo "- RAG 서비스: http://localhost:8002"
echo "- 멀티에이전트: http://localhost:8001"
echo "- 단일체인: http://localhost:8003"
echo "- Streamlit 앱: http://localhost:8501"
echo ""

echo "🔍 서비스 상태 확인:"
echo "curl http://localhost:8002/health  # RAG 서비스"
echo "curl http://localhost:8001/health  # 멀티에이전트"
echo "curl http://localhost:8003/health  # 단일체인"
echo ""

echo "⚠️  주의사항:"
echo "- 각 서비스는 별도의 터미널 탭에서 실행하세요"
echo "- RAG 서비스를 먼저 실행한 후 다른 서비스들을 실행하세요"
echo "- 서비스 종료는 Ctrl+C로 하세요"
