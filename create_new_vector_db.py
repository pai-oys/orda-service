import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

def normalize_data(item, item_type):
    """JSON 데이터를 정규화된 형식으로 변환"""
    tags = item.get("태그", "")
    if tags is None:
        tags = ""
        
    return {
        "id": str(item.get("id", item.get("이름", ""))) + "_" + item_type,
        "type": item_type,
        "title": item.get("이름", ""),
        "address": item.get("주소", "제주도"),
        "desc": item.get("소개", ""),
        "tags": tags.replace("#", "").replace(" ", "").split(",") if tags else [],
        "content": f"{item.get('이름', '')}은(는) {item.get('주소', '제주도')}에 위치한 {item_type} 장소입니다. {item.get('소개', '')} 관련 태그: {tags}"
    }

def load_json_file(file_path):
    """JSON 파일을 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_vector_db():
    """새로운 벡터 DB 생성 (384차원 임베딩 사용)"""
    
    print("🚀 새로운 벡터 DB 생성을 시작합니다...")
    
    # 1. 데이터 로딩
    print("📁 JSON 데이터를 로딩합니다...")
    try:
        food_data = [normalize_data(x, "음식") for x in load_json_file("visitjeju_food.json")]
        tour_data = [normalize_data(x, "관광") for x in load_json_file("visitjeju_tour.json")]
        hotel_data = [normalize_data(x, "숙소") for x in load_json_file("visitjeju_hotel.json")]
        event_data = [normalize_data(x, "행사") for x in load_json_file("visitjeju_event.json")]
        
        all_documents = food_data + tour_data + hotel_data + event_data
        print(f"✅ 총 {len(all_documents)}개의 문서를 로드했습니다.")
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return False
    except Exception as e:
        print(f"❌ 데이터 로딩 오류: {e}")
        return False

    # 2. 임베딩 모델 로드 (384차원)
    print("🤖 임베딩 모델을 로드합니다...")
    try:
        embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        print("✅ 임베딩 모델 로드 완료")
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        return False

    # 3. 텍스트 임베딩 생성
    print("🔢 텍스트 임베딩을 생성합니다...")
    try:
        texts = [doc["content"] for doc in all_documents]
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        print(f"✅ {len(embeddings)}개의 임베딩 생성 완료")
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}")
        return False

    # 4. ChromaDB 생성
    print("💾 ChromaDB를 생성합니다...")
    try:
        # 새로운 DB 경로
        new_db_path = "./new_vector_store"
        client = chromadb.PersistentClient(path=new_db_path)
        
        # 기존 컬렉션이 있다면 삭제
        collection_name = "jeju_place_info_384"
        try:
            existing_collection = client.get_collection(collection_name)
            client.delete_collection(collection_name)
            print(f"기존 컬렉션 '{collection_name}' 삭제됨")
        except:
            pass
        
        # 새 컬렉션 생성
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "제주도 여행 정보 (384차원 임베딩)"}
        )
        
        print(f"✅ 컬렉션 '{collection_name}' 생성 완료")
        
    except Exception as e:
        print(f"❌ ChromaDB 생성 실패: {e}")
        return False

    # 5. 데이터 저장
    print("📝 데이터를 저장합니다...")
    try:
        # 중복 제거
        unique_ids = []
        unique_texts = []
        unique_embeddings = []
        unique_metadatas = []
        seen_ids = set()
        
        for i, doc in enumerate(all_documents):
            doc_id = doc["id"]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_ids.append(doc_id)
                unique_texts.append(texts[i])
                unique_embeddings.append(embeddings[i].tolist())
                unique_metadatas.append({
                    "type": doc["type"],
                    "title": doc["title"],
                    "address": doc["address"]
                })
        
        print(f"중복 제거 후 {len(unique_ids)}개 문서")
        
        # 배치로 저장
        batch_size = 1000
        for i in range(0, len(unique_ids), batch_size):
            end_idx = min(i + batch_size, len(unique_ids))
            
            collection.add(
                documents=unique_texts[i:end_idx],
                embeddings=unique_embeddings[i:end_idx],
                metadatas=unique_metadatas[i:end_idx],
                ids=unique_ids[i:end_idx]
            )
            
            print(f"저장 진행: {end_idx}/{len(unique_ids)}")
        
        print(f"✅ 모든 데이터 저장 완료!")
        
    except Exception as e:
        print(f"❌ 데이터 저장 실패: {e}")
        return False

    # 6. 테스트 쿼리
    print("🔍 테스트 쿼리를 실행합니다...")
    try:
        test_results = collection.query(
            query_texts=["제주도 맛집 추천"],
            n_results=3
        )
        
        print("테스트 결과:")
        for i, doc in enumerate(test_results["documents"][0]):
            meta = test_results["metadatas"][0][i]
            print(f"  {i+1}. {meta.get('title')} ({meta.get('type')})")
            
    except Exception as e:
        print(f"❌ 테스트 쿼리 실패: {e}")
        return False

    print(f"""
🎉 새로운 벡터 DB 생성 완료!

📍 DB 경로: {new_db_path}
📍 컬렉션명: {collection_name}
📍 문서 수: {len(unique_ids)}
📍 임베딩 차원: 384

이제 smart_chatbot.py에서 다음과 같이 수정하세요:
1. db_path = "./new_vector_store"
2. 임베딩 모델: "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
""")
    
    return True

if __name__ == "__main__":
    create_vector_db()