# 🧠 AI Career Guidance Chatbot (RAG + Azure + Pinecone)

This is a full-stack AI chatbot project that integrates:
- **Azure OpenAI GPT models** for text generation
- **Pinecone** for semantic vector search (Retrieval-Augmented Generation)
- **Flask** backend for API endpoints
- **HTML/JS frontend** chat interface with document upload (PDF)
- **LangChain** for embeddings and RAG pipeline management

---

## 🚀 Features
✅ Chat interface powered by Azure OpenAI  
✅ PDF upload with vector storage in Pinecone  
✅ Toggle RAG (Retrieval-Augmented Generation) mode dynamically  
✅ Optionally use Hugging Face embeddings  
✅ Persistent chat history  
✅ Frontend with model display, status badges, and clean UI  

-----------------------------------------------------------------------

## 🧩 To Run Project:
Create & activate a virtual environment: 

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

Run The application -
python app.py

-----------------------------------------------------
Using the Chatbot
💬 Chat

Type your query and click “Send”

Toggle “Use RAG” to enable document retrieval

📄 Upload PDF

Click the “Upload” button

Your PDF will be split into text chunks → embedded → stored in Pinecone

Once uploaded, “RAG Enabled” badge appears

🔁 Toggle RAG

Click Switch RAG to enable or disable vector-based retrieval dynamically


