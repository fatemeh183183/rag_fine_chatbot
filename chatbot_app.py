import streamlit as st
import openai
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load OpenAI API key
st.set_page_config(page_title="🧠 RAG + Fine-Tuning Chatbot")
st.title("🤖 AI Support Chatbot (Fine-Tuned + RAG + Prompt)")

# User inputs for keys and model
api_key = st.text_input("🔑 OpenAI API Key", type="password")
fine_tuned_model = st.text_input("🤖 Fine-Tuned Model Name (e.g. ft:gpt-3.5-turbo:xyz)")

# Load embeddings and FAISS index (simulating a knowledge base)
@st.cache_resource
def load_vector_store():
    with open("rag_faq.pkl", "rb") as f:
        data = pickle.load(f)
    return data["index"], data["texts"], data["model"]

# Get top relevant chunk from FAISS index
def retrieve_context(query, index, texts, embed_model):
    query_vec = embed_model.encode([query])
    D, I = index.search(query_vec, k=1)  # top-1 most similar
    return texts[I[0][0]]

# Generate answer using fine-tuned GPT with prompt
def generate_answer(prompt, model_name, api_key):
    openai.api_key = api_key
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a friendly, concise customer support assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Main chatbot logic
question = st.text_input("💬 Ask your question")

if api_key and fine_tuned_model and question:
    with st.spinner("Thinking..."):
        index, texts, embed_model = load_vector_store()
        context = retrieve_context(question, index, texts, embed_model)

        # Prompt template
        full_prompt = f"""Use the following context to answer like a customer support agent:

Context: {context}

Question: {question}"""
        answer = generate_answer(full_prompt, fine_tuned_model, api_key)
        st.success("✅ Answer:")
        st.write(answer)
