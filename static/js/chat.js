class ChatInterface {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendMessage');
        this.useRAG = document.getElementById('useRAG');
        this.clearChatBtn = document.getElementById('clearChat');
        this.toggleRAGBtn = document.getElementById('toggleRAG');
        this.currentModel = document.getElementById('currentModel');
        this.ragStatus = document.getElementById('ragStatus');
        
        this.initializeEventListeners();
        this.loadSessionInfo();
    }
    
    initializeEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        this.toggleRAGBtn.addEventListener('click', () => this.toggleRAG());
        
        // Document upload
        const documentUpload = document.getElementById('chatDocumentUpload');
        documentUpload.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.uploadDocument(e.target.files[0]);
            }
        });
    }
    
    loadSessionInfo() {
        this.currentModel.textContent = 'Azure OpenAI';
        this.ragStatus.textContent = 'Disabled';
        this.ragStatus.className = 'badge bg-secondary';
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.showTypingIndicator();
        
        try {
            const response = await fetch('/api/send-message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    use_rag: this.useRAG.checked
                })
            });

            if (!response.ok) throw new Error('Server error while sending message.');
            const data = await response.json();

            this.hideTypingIndicator();
            if (data.success) {
                this.addMessage('assistant', data.response);
            } else {
                this.addMessage('system', `Error: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('system', `Network error: ${error.message}`);
        }
    }
    
    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant-message';
        typingDiv.id = 'typingIndicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'typing-indicator';
        contentDiv.innerHTML = 'AI is thinking <span class="typing-dots"></span>';
        
        typingDiv.appendChild(contentDiv);
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) typingIndicator.remove();
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    async clearChat() {
        try {
            const response = await fetch('/api/clear-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) throw new Error('Server error clearing chat.');
            const data = await response.json();

            if (data.success) {
                this.chatMessages.innerHTML = `
                    <div class="text-center text-muted my-4">
                        <i class="fas fa-robot fa-2x mb-2"></i>
                        <p>Start a conversation with your AI assistant!</p>
                    </div>
                `;
            } else {
                alert('Error clearing chat: ' + (data.error || 'Unknown error.'));
            }
        } catch (error) {
            alert('Error clearing chat: ' + error.message);
        }
    }
    
    async toggleRAG() {
        try {
            const currentStatus = this.ragStatus.textContent === 'Enabled';
            const response = await fetch('/api/toggle-rag', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enable_rag: !currentStatus })
            });

            if (!response.ok) throw new Error('Server error toggling RAG.');
            const data = await response.json();

            if (data.success) {
                this.ragStatus.textContent = !currentStatus ? 'Enabled' : 'Disabled';
                this.ragStatus.className = !currentStatus ? 'badge bg-success' : 'badge bg-secondary';
                this.toggleRAGBtn.innerHTML = !currentStatus 
                    ? '<i class="fas fa-toggle-on"></i> Switch RAG' 
                    : '<i class="fas fa-toggle-off"></i> Switch RAG';
            } else {
                alert('Error toggling RAG: ' + (data.error || 'Unknown error.'));
            }
        } catch (error) {
            alert('Error toggling RAG: ' + error.message);
        }
    }
    
    async uploadDocument(file) {
        const uploadStatus = document.getElementById('chatUploadStatus');
        const formData = new FormData();
        formData.append('file', file);
        
        uploadStatus.innerHTML = '<span class="text-info">Uploading...</span>';
        
        try {
            const response = await fetch('/api/upload-document', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);
            
            let data;
            try {
                data = await response.json();
            } catch {
                throw new Error('Invalid JSON response from server.');
            }

            if (data.success) {
                uploadStatus.innerHTML = '<span class="text-success">Document uploaded successfully!</span>';
                this.ragStatus.textContent = 'Enabled';
                this.ragStatus.className = 'badge bg-success';
            } else {
                uploadStatus.innerHTML = `<span class="text-danger">Upload failed: ${data.error || 'Unknown error'}</span>`;
            }
        } catch (error) {
            uploadStatus.innerHTML = `<span class="text-danger">Upload error: ${error.message}</span>`;
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ChatInterface();
});
