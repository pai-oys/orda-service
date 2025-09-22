import chromadb

# vector_store 폴더의 ChromaDB 확인
try:
    client = chromadb.PersistentClient(path="./vector_store")
    collections = client.list_collections()
    
    print("=== ChromaDB 상태 확인 ===")
    print(f"DB 경로: ./vector_store")
    print(f"발견된 컬렉션 수: {len(collections)}")
    
    if collections:
        print("\n컬렉션 목록:")
        for collection in collections:
            print(f"- 이름: {collection.name}")
            print(f"- 문서 수: {collection.count()}")
            print()
    else:
        print("컬렉션이 없습니다!")
        
except Exception as e:
    print(f"오류: {e}")