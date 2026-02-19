import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the wrapped sentence-transformer model
MODEL_PATH = "models/learnora_finetuned_stv2"
model = SentenceTransformer(MODEL_PATH)

# Load the metadata (enriched dataset)
with open("./datasets/learnora_metadata_final.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Load the FAISS index
index = faiss.read_index("./datasets/learnora_faiss_final.index")

# Semantic search function
def search(query, top_k=5):
    query_embedding = model.encode(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        result = dataset[idx].copy()
        result["similarity_score"] = float(score)
        results.append(result)
    
    return results

# For command line or script testing
if __name__ == "__main__":
    query = input("Enter your query: ")
    results = search(query)

    print("\nTop Results:\n")
    for i, res in enumerate(results, 1):
        print(f"{i}. Title: {res['title']}")
        print(f"   Source: {res['source']}")
        print(f"   Labels: {', '.join(res['labels'])}")
        print(f"   Link: {res['link']}")
        print(f"   Credibility score: {res['credibility_score']}")
        if 'confidence' in res:
            print(f"   Confidence: {res['confidence']}")
        #print(f"   Similarity Score: {res['similarity_score']:.2f}")
        #print("-" * 60)
