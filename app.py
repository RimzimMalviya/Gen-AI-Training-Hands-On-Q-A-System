from flask import Flask, render_template, request
from utils.chat_utils import get_chat_model
from utils.rag_utils import store_pdf_in_pinecone, query_rag
from config import *

app = Flask(__name__)

memory_chain = None
messages = []

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global memory_chain, messages
    model = request.form.get("model")
    system_message = request.form.get("system_message", "")
    enable_rag = "enable_rag" in request.form

    if "pdf" in request.files and enable_rag:
        pdf = request.files["pdf"]
        if pdf.filename:
            path = f"./uploads/{pdf.filename}"
            pdf.save(path)
            msg = store_pdf_in_pinecone(path, PINECONE_INDEX, PINECONE_API_KEY)
            messages.append({"role": "bot", "text": msg})

    if not memory_chain:
        memory_chain = get_chat_model(model, system_message)

    user_input = request.form.get("user_input")
    if user_input:
        if enable_rag:
            rag_context = query_rag(user_input, PINECONE_INDEX, PINECONE_API_KEY)
            user_input += f"\nContext:\n{rag_context}"

        response = memory_chain.run(input=user_input)
        messages.append({"role": "user", "text": user_input})
        messages.append({"role": "bot", "text": response})

    return render_template("chat.html", messages=messages, model=model, system_message=system_message, enable_rag=enable_rag)

if __name__ == "__main__":
    app.run(debug=True)
