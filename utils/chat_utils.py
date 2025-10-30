from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import tiktoken
from config import Config
import requests
import json
import time

class ChatManager:
    def __init__(self, config):
        self.config = config
        self.azure_llm = None
        self.huggingface_client = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize both Azure and Hugging Face models"""
        try:
            # Initialize Azure OpenAI
            if self.config.AZURE_OPENAI_API_KEY and self.config.AZURE_OPENAI_ENDPOINT:
                self.azure_llm = AzureChatOpenAI(
                    azure_deployment=self.config.AZURE_DEPLOYMENT_NAME,
                    openai_api_version=self.config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
                    api_key=self.config.AZURE_OPENAI_API_KEY,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS
                )
                print("Azure OpenAI initialized successfully")
            else:
                print("Azure OpenAI not configured properly")
                
        except Exception as e:
            print(f"Error initializing Azure OpenAI: {e}")
        
        try:
            # Initialize Hugging Face with direct API client
            if self.config.HUGGINGFACEHUB_API_TOKEN:
                self.huggingface_client = HuggingFaceDirectClient(
                    api_token=self.config.HUGGINGFACEHUB_API_TOKEN,
                    model_name=self.config.HUGGINGFACE_MODEL,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS
                )
                print("Hugging Face (Direct API) initialized successfully")
        except Exception as e:
            print(f"Error initializing Hugging Face: {e}")
    
    def get_chat_response(self, model_choice, messages):
        """Get chat response from selected model"""
        if model_choice == 'azure' and self.azure_llm:
            return self._get_azure_response(messages)
        elif model_choice == 'huggingface' and self.huggingface_client:
            return self._get_huggingface_response(messages)
        else:
            available_models = []
            if self.azure_llm:
                available_models.append('azure')
            if self.huggingface_client:
                available_models.append('huggingface')
            raise Exception(f"Selected model '{model_choice}' is not available. Available: {available_models}")
    
    def _get_azure_response(self, messages):
        """Get response from Azure OpenAI"""
        try:
            # Convert messages to LangChain format
            lc_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    lc_messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    lc_messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    lc_messages.append(AIMessage(content=msg['content']))
            
            response = self.azure_llm.invoke(lc_messages)
            return response.content
        
        except Exception as e:
            raise Exception(f"Azure OpenAI error: {str(e)}")
    
    def _get_huggingface_response(self, messages):
        """Get response from Hugging Face using direct API"""
        try:
            # Build conversation prompt
            conversation_text = self._build_conversation_text(messages)
            response = self.huggingface_client.generate_response(conversation_text)
            return response
            
        except Exception as e:
            raise Exception(f"Hugging Face error: {str(e)}")
    
    def _build_conversation_text(self, messages):
        """Build conversation text for Hugging Face models"""
        conversation = ""
        for msg in messages:
            if msg['role'] == 'system':
                conversation += f"System: {msg['content']}\n\n"
            elif msg['role'] == 'user':
                conversation += f"User: {msg['content']}\n\n"
            elif msg['role'] == 'assistant':
                conversation += f"Assistant: {msg['content']}\n\n"
        
        conversation += "Assistant:"
        return conversation
    
    def get_rag_response(self, model_choice, query, chat_history, system_message, rag_context):
        """Get RAG-enhanced response with document context"""
        # Build enhanced system message with RAG context
        enhanced_system_message = system_message
        if rag_context:
            enhanced_system_message += f"\n\nAdditional context from uploaded documents:\n{rag_context}\n\nPlease use this context to provide more accurate and relevant responses. If the context doesn't contain relevant information, use your general knowledge."

        messages = [
            {'role': 'system', 'content': enhanced_system_message},
            *chat_history,
            {'role': 'user', 'content': query}
        ]
        return self.get_chat_response(model_choice, messages)
    
    def is_azure_available(self):
        return self.azure_llm is not None
    
    def is_huggingface_available(self):
        return self.huggingface_client is not None


class HuggingFaceDirectClient:
    """Direct Hugging Face API client that actually works"""
    
    def __init__(self, api_token, model_name, temperature=0.7, max_tokens=1000):
        self.api_token = api_token
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, text):
        """Generate response using Hugging Face Inference API"""
        try:
            # Prepare the payload
            payload = {
                "inputs": text,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "do_sample": True,
                    "return_full_text": False,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": True
                }
            }
            
            print(f"Sending request to Hugging Face model: {self.model_name}")
            
            # Make the API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=120  # Longer timeout for model loading
            )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                return self._extract_response_text(result)
            elif response.status_code == 503:
                # Model is loading, wait and retry
                print("Model is loading, waiting 10 seconds...")
                time.sleep(10)
                return self.generate_response(text)  # Retry
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(f"{error_msg}")
                return f"Error: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timeout - model is taking too long to respond"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _extract_response_text(self, result):
        """Extract response text from Hugging Face API response"""
        try:
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    return result[0]['generated_text'].strip()
                else:
                    return str(result[0]).strip()
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text'].strip()
            else:
                return str(result).strip()
        except:
            return "No response generated"