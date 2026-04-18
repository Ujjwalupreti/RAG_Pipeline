from src.generation import get_qa_chain
from src.ingestion import ingest_documents
import os

def main():
    if not os.path.exists("./vector_store") or not os.listdir("./vector_store"):
        print("Initial setup: Processing documents...")
        ingest_documents()

    print("Initializing Advanced RAG Pipeline...")
    qa_chain = get_qa_chain()
    
    print("\n--- Advanced RAG System Active ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        query = input("Ask a question: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        print("\nSearching and Re-ranking (Hybrid Search active)...")
        response = qa_chain.invoke({"input": query})
        
        print("\n--- Answer ---")
        print(response["answer"])
        print("\n--- Source Documents Used ---")
        for i, doc in enumerate(response["context"]):
            print(f"[{i+1}] {doc.page_content[:150]}...")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()
