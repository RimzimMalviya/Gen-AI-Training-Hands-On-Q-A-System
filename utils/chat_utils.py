from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def get_chat_model(provider, system_message):
    if provider == "azure":
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            openai_api_version="2024-02-15-preview"
        )
    else:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="text-generation",
            max_new_tokens=512
        )
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory, verbose=False)
    if system_message:
        chain.memory.save_context({"input": ""}, {"output": system_message})
    return chain
