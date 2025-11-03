## 🧩 To Run Project:
Create & activate a virtual environment: 

python3.11 -m venv venv
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


