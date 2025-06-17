# 🤖 Fine-Tuned RAG-Powered Customer Support Chatbot

This project demonstrates how to combine:
- ✅ Fine-Tuning (custom Q&A tone/behavior)
- 🔍 RAG (Retrieval-Augmented Generation for live knowledge)
- 🧠 Prompt Engineering (custom system instructions)

## 📦 Files

- `chatbot_app.py` — Main chatbot app (Streamlit)
- `build_rag_index.py` — Script to create RAG vector store
- `rag_faq.pkl` — Generated knowledge base (run script first)
- `requirements.txt` — Dependencies
- `README.md` — This guide

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Build the vector store once:

```bash
python build_rag_index.py
```

3. Run the chatbot:

```bash
streamlit run chatbot_app.py
```

4. Enter your:
- 🔑 OpenAI API Key
- 🤖 Fine-tuned model name (e.g. `ft:gpt-3.5-turbo:xyz`)
- Ask a customer support question!

---

## 📚 How It Works

- User asks a question
- App retrieves the most relevant FAQ chunk using **FAISS**
- Merges it into a **prompt template**
- Sends it to your **fine-tuned GPT model** for accurate + styled response


