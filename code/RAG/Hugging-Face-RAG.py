from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# 1. Documents
documents = [
    "LangChain is a framework to build applications with LLMs.",
    "RAG enhances LLMs by providing external knowledge.",
    "Hugging Face hosts transformers and datasets for NLP."
]

# 2. Embed documents
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents)

# 3. Build FAISS index
index = faiss.IndexFlatL2(doc_embeddings[0].shape[0])
index.add(np.array(doc_embeddings))

# 4. Embed user query
query = "How does RAG help language models?"
query_embedding = embedder.encode([query])

# 5. Search
D, I = index.search(np.array(query_embedding), k=2)
retrieved_docs = [documents[i] for i in I[0]]

# 6. Build context
context = "\n".join(retrieved_docs)

# 7. Use a generator model
generator = pipeline("text-generation", model="gpt2")
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
response = generator(prompt, max_new_tokens=50, do_sample=True)
print(response[0]["generated_text"])
