import chromadb

# 기존 vector_store의 컬렉션 확인
try:
    print("=== 기존 ChromaDB 확인 ===")
    client = chromadb.PersistentClient(path="./vector_store")
    
    # 모든 컬렉션 나열
    collections = client.list_collections()
    print(f"총 컬렉션 수: {len(collections)}")
    
    for i, collection in enumerate(collections):
        print(f"\n컬렉션 {i+1}:")
        print(f"  - 이름: '{collection.name}'")
        print(f"  - 문서 수: {collection.count()}")
        
        # 첫 번째 문서 샘플 확인
        if collection.count() > 0:
            sample = collection.peek(limit=1)
            print(f"  - 샘플 ID: {sample['ids'][0] if sample['ids'] else 'None'}")
            if sample.get('metadatas') and sample['metadatas'][0]:
                print(f"  - 샘플 메타데이터: {sample['metadatas'][0]}")
            if sample.get('documents') and sample['documents'][0]:
                print(f"  - 샘플 문서: {sample['documents'][0][:100]}...")
                
except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()