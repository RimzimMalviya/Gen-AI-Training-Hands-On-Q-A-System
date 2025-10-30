from flask import Flask, render_template, request, jsonify, session
import os
from werkzeug.utils import secure_filename
from config import Config
import uuid

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('documents', exist_ok=True)

from flask import Flask, render_template, request, jsonify, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import time

app = Flask(__name__)
app.config.from_object(Config)

# Initialize rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)


# Initialize managers with error handling
try:
    from utils.chat_utils import ChatManager
    from utils.rag_utils import SimpleRAGManager  # Use simple RAG
    
    chat_manager = ChatManager(Config)
    rag_manager = SimpleRAGManager(Config)  # Simple RAG without embeddings
    
    print("All managers initialized successfully!")
    
except Exception as e:
    print(f"Error during initialization: {e}")
    chat_manager = None
    rag_manager = None
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all utility files are present.")
    chat_manager = None
    rag_manager = None
except Exception as e:
    print(f" Error initializing managers: {e}")
    chat_manager = None
    rag_manager = None

# Routes
@app.route('/')
def index():
    """Main configuration page"""
    config_status = {
        'azure_configured': bool(Config.AZURE_OPENAI_API_KEY and Config.AZURE_OPENAI_ENDPOINT),
        'huggingface_configured': bool(Config.HUGGINGFACEHUB_API_TOKEN),
        'managers_initialized': bool(chat_manager)
    }
    return render_template('index.html', config_status=config_status)

@app.route('/chat')
def chat():
    """Chat interface page"""
    return render_template('chat.html')

@limiter.limit("20 per minute")
@app.route('/api/initialize-chat', methods=['POST'])
def initialize_chat():
    try:
        if not chat_manager:
            return jsonify({
                'success': False,
                'error': 'Chat manager not initialized. Please check your configuration.'
            }), 500
        
        data = request.get_json()
        model_choice = data.get('model_choice', 'azure')
        system_message = data.get('system_message', 'You are a helpful AI assistant.')
        
        # Validate model choice
        if model_choice == 'azure' and not Config.AZURE_OPENAI_API_KEY:
            return jsonify({
                'success': False,
                'error': 'Azure OpenAI is not configured.'
            }), 400
        
        if model_choice == 'huggingface' and not Config.HUGGINGFACEHUB_API_TOKEN:
            return jsonify({
                'success': False,
                'error': 'Hugging Face is not configured.'
            }), 400
        
        # Generate unique chat ID
        chat_id = str(uuid.uuid4())
        
        # Initialize chat session
        session['chat_id'] = chat_id
        session['model_choice'] = model_choice
        session['system_message'] = system_message
        session['messages'] = []
        
        # Add system message to history
        session['messages'].append({
            'role': 'system',
            'content': system_message
        })
        
        return jsonify({
            'success': True,
            'chat_id': chat_id,
            'message': 'Chat initialized successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@limiter.limit("10 per minute")
@app.route('/api/send-message', methods=['POST'])
def send_message():
    try:
        if not chat_manager:
            return jsonify({
                'success': False,
                'error': 'Chat manager not initialized'
            }), 500
        
        data = request.get_json()
        user_message = data.get('message', '')
        use_rag = data.get('use_rag', False)
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400
        
        # Get session data
        chat_id = session.get('chat_id')
        model_choice = session.get('model_choice', 'azure')
        system_message = session.get('system_message', '')
        messages = session.get('messages', [])
        
        if not chat_id:
            return jsonify({
                'success': False,
                'error': 'Chat not initialized'
            }), 400
        
        # Add user message to history
        messages.append({
            'role': 'user',
            'content': user_message
        })
        
        # Get response from chatbot
        rag_context = ""
        used_rag = False
        
        if use_rag and rag_manager and rag_manager.is_rag_enabled():
            print("Searching documents for RAG context...")
            # Get relevant document context
            rag_context_list = rag_manager.query_documents(user_message, k=3)
            rag_context = "\n\n".join(rag_context_list) if rag_context_list else ""
            
            if rag_context:
                print(f"Found RAG context: {len(rag_context_list)} chunks")
                response = chat_manager.get_rag_response(
                    model_choice=model_choice,
                    query=user_message,
                    chat_history=messages[:-1],  # Exclude current user message
                    system_message=system_message,
                    rag_context=rag_context
                )
                used_rag = True
                print("Response generated with RAG context")
            else:
                print("No relevant RAG context found, using regular chat")
                response = chat_manager.get_chat_response(
                    model_choice=model_choice,
                    messages=messages
                )
        else:
            # Use regular chat response
            print("Using regular chat (RAG disabled)")
            response = chat_manager.get_chat_response(
                model_choice=model_choice,
                messages=messages
            )
        
        # Add assistant response to history
        messages.append({
            'role': 'assistant',
            'content': response
        })
        
        # Update session
        session['messages'] = messages
        
        return jsonify({
            'success': True,
            'response': response,
            'chat_id': chat_id,
            'used_rag': used_rag
        })
    
    except Exception as e:
        print(f"Error in send-message: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    try:
        if not rag_manager:
            return jsonify({
                'success': False,
                'error': 'RAG manager not initialized'
            }), 500
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process and index the document
            success = rag_manager.process_document(filepath)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Document uploaded and processed successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to process document'
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Only PDF files are supported'
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/toggle-rag', methods=['POST'])
def toggle_rag():
    try:
        if not rag_manager:
            return jsonify({
                'success': False,
                'error': 'RAG manager not initialized'
            }), 500
        
        data = request.get_json()
        enable_rag = data.get('enable_rag', False)
        
        rag_manager.set_rag_enabled(enable_rag)
        
        return jsonify({
            'success': True,
            'message': f'RAG {"enabled" if enable_rag else "disabled"}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    try:
        # Reset session messages but keep configuration
        system_message = session.get('system_message', '')
        session['messages'] = [{'role': 'system', 'content': system_message}]
        
        return jsonify({
            'success': True,
            'message': 'Chat cleared successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'flask': 'running',
        'chat_manager': 'initialized' if chat_manager else 'not_initialized',
        'rag_manager': 'initialized' if rag_manager else 'not_initialized',
        'azure_openai': 'configured' if Config.AZURE_OPENAI_API_KEY else 'not_configured',
        'huggingface': 'configured' if Config.HUGGINGFACEHUB_API_TOKEN else 'not_configured',
        'pinecone': 'configured' if Config.PINECONE_API_KEY else 'not_configured'
    }
    return jsonify(status)

@app.route('/debug-routes')
def debug_routes():
    """Show all available routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':  # Exclude static files
            routes.append({
                'endpoint': rule.endpoint,
                'methods': list(rule.methods),
                'path': str(rule)
            })
    return jsonify(routes)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_routes': [
            '/',
            '/chat', 
            '/health',
            '/debug-routes',
            '/api/initialize-chat',
            '/api/send-message',
            '/api/upload-document',
            '/api/toggle-rag',
            '/api/clear-chat'
        ]
    }), 404

if __name__ == '__main__':
    print("Starting AI Chatbot Dashboard...")
    print("Available routes:")
    print("  - / (main page)")
    print("  - /chat (chat interface)")
    print("  - /health (status check)")
    print("  - /debug-routes (list all routes)")
    
    if not chat_manager:
        print("\n WARNING: Chat manager not initialized!")
    else:
        print("All systems ready!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

@app.route('/api/rag-status')
def rag_status():
    """Check RAG status and document info"""
    try:
        if rag_manager:
            doc_info = rag_manager.get_document_info()
            return jsonify({
                'success': True,
                'rag_enabled': rag_manager.is_rag_enabled(),
                'documents_loaded': doc_info['chunks_loaded'] > 0,
                'chunks_loaded': doc_info['chunks_loaded'],
                'rag_ready': rag_manager.is_rag_enabled()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'RAG manager not initialized'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'success': False,
        'error': f'Rate limit exceeded: {e.description}'
    }), 429

@app.route('/api/test-huggingface')
def test_huggingface():
    """Test Hugging Face connection"""
    try:
        if not chat_manager or not chat_manager.huggingface_client:
            return jsonify({
                'success': False,
                'error': 'Hugging Face client not initialized'
            })
        
        # Test with a simple prompt
        test_prompt = "Hello, how are you?"
        response = chat_manager.huggingface_client.generate_response(test_prompt)
        
        return jsonify({
            'success': True,
            'model': Config.HUGGINGFACE_MODEL,
            'test_prompt': test_prompt,
            'response': response,
            'status': 'working' if 'Error' not in response else 'error'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    
