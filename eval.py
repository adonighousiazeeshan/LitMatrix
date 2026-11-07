import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from evaldataset import GOLD_BENCHMARK
from extractor import extract_paper_metadata

chroma_dir = "./chroma_db"

def run_eval():
    if not os.path.exists(chroma_dir):
        print("⚠️ Please upload your documents first.")
        return
        
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retrieval_success = 0
    total_que = len(GOLD_BENCHMARK)
    
    vector_store1 = Chroma(persist_directory=chroma_dir, embedding_function=embedding)
    retriever = vector_store1.as_retriever(search_type='mmr', search_kwargs={'k': 6, 'fetch_k': 15})
    
    print("--- Test 1: Evaluating Retriever ---")
    for i, test_case in enumerate(GOLD_BENCHMARK, 1):
        query = test_case['question']
        file_name = test_case['target_filename']  # <-- Uses target_filename correctly
        
        retrieved_docs = retriever.invoke(query)
        
        # Clean check for filename match
        found = any(
            doc.metadata.get("source", "").replace("\\", "/").split("/")[-1] == file_name 
            for doc in retrieved_docs
        )
        
        if found:
            retrieval_success += 1
        else:
            print(f"  ❌ Test {i} (Retriever): Query failed -> Target file '{file_name}' not found in top chunks.")
            
    retrieval_recall = (retrieval_success / total_que) * 100
    print(f"\n📊 Retrieval Recall Rate: {retrieval_recall:.1f}% ({retrieval_success}/{total_que})\n")

    # --- TEST 2: Evaluating Structured Extraction Module (Pydantic Schema) ---
    print("--- Test 2: Evaluating Pydantic Extractor ---")
    files = list(set(tc["target_filename"] for tc in GOLD_BENCHMARK))  # <-- Uses target_filename correctly
    try:
        extraction_res = extract_paper_metadata(vector_store1, files)
        print(f"  ✅ Extractor Test Passed: Successfully generated structured schema for {len(extraction_res)} paper(s).")
        for res in extraction_res:
            print(f"     - Paper: {res['Paper']} | Algorithm Extracted: {res['Algorithm / Model']}")
    except Exception as e:
        print(f"  ❌ Extractor Test Failed with error: {str(e)}")

if __name__ == "__main__":
    run_eval()