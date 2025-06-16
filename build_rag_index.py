import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Sample support FAQ entries
texts = [
    "You can reset your password by clicking 'Forgot Password' on the login page.",
    "We offer a full refund within 30 days of purchase.",
    "You can upgrade or downgrade your subscription anytime from your account settings.",
    "Track your order from your dashboard under 'My Orders'.",
    "To cancel your account, go to Account Settings > Cancel Subscription."
]

# Embed model (lightweight)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and metadata
with open("rag_faq.pkl", "wb") as f:
    pickle.dump({"index": index, "texts": texts, "model": model}, f)

print("✅ Knowledge base saved as rag_faq.pkl")
