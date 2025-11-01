from flask import Flask, render_template, request, jsonify, session
from config import Config
from utils.chat_utils import ChatManager
from werkzeug.utils import secure_filename
import os, uuid

app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

chat_manager = ChatManager(Config)
from utils.rag_utils import PineconeRAGManager
rag_manager = PineconeRAGManager(Config)

available_models = [
    {"id": "azure", "name": "Azure OpenAI"},
    {"id": "huggingface", "name": "Hugging Face"},
]


@app.route("/")
def index():
    from config import Config
    config_status = {
        "azure_configured": bool(Config.AZURE_OPENAI_API_KEY and Config.AZURE_OPENAI_ENDPOINT),
        "huggingface_configured": bool(Config.HUGGINGFACEHUB_API_TOKEN),
    }
    return render_template("index.html", config_status=config_status)



@app.route('/chat')
def chat():
    return render_template("chat.html")

@app.route('/api/initialize-chat', methods=['POST'])
def initialize_chat():
    data = request.get_json()
    model = data.get("model_choice")
    sys_msg = data.get("system_message", "You are a helpful assistant.")
    chat_id = str(uuid.uuid4())
    session["chat_id"] = chat_id
    session["model_choice"] = model
    session["system_message"] = sys_msg
    session["messages"] = [{"role": "system", "content": sys_msg}]
    return jsonify({"success": True, "chat_id": chat_id})

@app.route('/api/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    msg = data.get("message", "")
    use_rag = data.get("use_rag", False)
    model_choice = session.get("model_choice")
    messages = session.get("messages", [])

    messages.append({"role": "user", "content": msg})

    rag_context = ""
    if use_rag and rag_manager.is_rag_enabled():
        ctx = rag_manager.query_documents(msg)
        if ctx:
            rag_context = "\n\n".join(ctx[:3])
            sys_msg = session["system_message"] + f"\n\nContext:\n{rag_context}"
            messages.insert(0, {"role": "system", "content": sys_msg})

    response = chat_manager.get_chat_response(model_choice, messages)
    messages.append({"role": "assistant", "content": response})
    session["messages"] = messages
    return jsonify({"success": True, "response": response})

@app.route('/api/upload-document', methods=['POST'])
def upload_doc():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file"}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"success": False, "error": "PDF only"}), 400
    path = os.path.join(Config.UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)
    ok = rag_manager.process_document(path)
    return jsonify({"success": ok})

@app.route('/api/toggle-rag', methods=['POST'])
def toggle_rag():
    enable = request.get_json().get("enable_rag", False)
    rag_manager.set_rag_enabled(enable)
    return jsonify({"success": True, "enabled": enable})

if __name__ == "__main__":
    app.run(debug=True)
