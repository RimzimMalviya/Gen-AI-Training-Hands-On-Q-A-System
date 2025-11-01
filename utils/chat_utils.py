from openai import AzureOpenAI
from config import Config
import requests, time

class ChatManager:
    def __init__(self, config):
        self.config = config
        self.azure_client = None
        self.hf_client = None
        self._init_models()

    def _init_models(self):
        try:
            if self.config.AZURE_OPENAI_API_KEY and self.config.AZURE_OPENAI_ENDPOINT:
                self.azure_client = AzureOpenAI(
                    api_key=self.config.AZURE_OPENAI_API_KEY,
                    azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
                    api_version=self.config.AZURE_OPENAI_API_VERSION
)
                print("Azure client ready")
        except Exception as e:
            print(f"Azure init error: {e}")

        try:
            if self.config.HUGGINGFACEHUB_API_TOKEN:
                self.hf_client = HuggingFaceClient(self.config)
                print("Hugging Face client ready")
        except Exception as e:
            print(f"HF init error: {e}")

    def get_chat_response(self, model_choice, messages):
        if model_choice == "azure" and self.azure_client:
            return self._azure_response(messages)
        elif model_choice == "huggingface" and self.hf_client:
            return self.hf_client.chat(messages)
        else:
            return "Error: model not available"

    def _azure_response(self, messages):
        formatted = [{"role": m["role"], "content": m["content"]} for m in messages]
        res = self.azure_client.chat.completions.create(
            model=self.config.AZURE_DEPLOYMENT_NAME,  # still your deployment name
            messages=formatted,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS
        )
        return res.choices[0].message.content


class HuggingFaceClient:
    def __init__(self, config):
        self.api_url = f"https://api-inference.huggingface.co/models/{config.HUGGINGFACE_MODEL}"
        self.headers = {"Authorization": f"Bearer {config.HUGGINGFACEHUB_API_TOKEN}"}
        self.max_tokens = config.MAX_TOKENS
        self.temp = config.TEMPERATURE

    def chat(self, messages):
        prompt = ""
        for m in messages:
            prompt += f"{m['role'].capitalize()}: {m['content']}\n"
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": self.temp, "max_new_tokens": self.max_tokens}
        }
        r = requests.post(self.api_url, headers=self.headers, json=payload)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"]
        return f"Error: HF returned {r.status_code}"
