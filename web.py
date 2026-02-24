#!/usr/bin/env python3
"""
SLO Web Interface
Simple web UI for interacting with SLO cognitive system

Usage:
    python web.py
    # Opens http://localhost:5000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template_string, request, jsonify
from domains.soul.cognitive import CognitiveSLO
from domains.soul.foundation import SLOConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("slo.web")

# Create Flask app
app = Flask(__name__)

# Initialize SLO
slo = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SLO - Self-Learning Organism</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #30363d;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #58a6ff, #8b949e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { color: #8b949e; font-size: 0.9em; }
        
        .chat-container {
            background: #161b22;
            border-radius: 12px;
            border: 1px solid #30363d;
            height: 60vh;
            display: flex;
            flex-direction: column;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 8px;
            max-width: 80%;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            background: #1f6feb;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        
        .message.slo {
            background: #21262d;
            border: 1px solid #30363d;
            border-bottom-left-radius: 4px;
        }
        
        .message .label {
            font-size: 0.7em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
            opacity: 0.7;
        }
        
        .input-area {
            padding: 15px;
            border-top: 1px solid #30363d;
            display: flex;
            gap: 10px;
        }
        
        input {
            flex: 1;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #30363d;
            background: #0d1117;
            color: #c9d1d9;
            font-size: 1em;
            outline: none;
        }
        input:focus { border-color: #58a6ff; }
        
        button {
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
            background: #238636;
            color: white;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #2ea043; }
        button:disabled { background: #30363d; cursor: not-allowed; }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .stat-value { font-size: 1.5em; font-weight: bold; color: #58a6ff; }
        .stat-label { font-size: 0.8em; color: #8b949e; }
        
        .systems {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 20px;
            padding: 15px;
            background: #161b22;
            border-radius: 8px;
        }
        
        .system-tag {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            background: #30363d;
            color: #8b949e;
        }
        
        .loading {
            display: inline-block;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 SLO</h1>
            <div class="subtitle">Self-Learning Organism</div>
        </header>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="knowledge-count">0</div>
                <div class="stat-label">Knowledge</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="sessions">0</div>
                <div class="stat-label">Sessions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="systems">16</div>
                <div class="stat-label">Systems</div>
            </div>
        </div>
        
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message slo">
                    <div class="label">SLO</div>
                    Hello! I'm SLO, a self-learning organism with cognitive capabilities. How can I help you today?
                </div>
            </div>
            
            <div class="input-area">
                <input type="text" id="user-input" placeholder="Ask me anything..." autofocus>
                <button id="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div class="systems">
            <span class="system-tag">Memory</span>
            <span class="system-tag">RAG</span>
            <span class="system-tag">Emotional IQ</span>
            <span class="system-tag">Metacognition</span>
            <span class="system-tag">Hebbian Learning</span>
            <span class="system-tag">Knowledge Graph</span>
        </div>
    </div>
    
    <script>
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        function addMessage(content, isUser) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user' : 'slo'}`;
            div.innerHTML = `<div class="label">${isUser ? 'You' : 'SLO'}</div>${content}`;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        async function sendMessage() {
            const text = input.value.trim();
            if (!text) return;
            
            addMessage(text, true);
            input.value = '';
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<span class="loading">Thinking...</span>';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                
                const data = await response.json();
                
                // Update stats
                document.getElementById('knowledge-count').textContent = data.knowledge || 0;
                document.getElementById('sessions').textContent = (parseInt(document.getElementById('sessions').textContent) + 1);
                
                addMessage(data.response, false);
            } catch (e) {
                addMessage('Error: ' + e.message, false);
            }
            
            sendBtn.disabled = false;
            sendBtn.innerHTML = 'Send';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    global slo
    
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Initialize SLO if needed
        if slo is None:
            logger.info("Initializing SLO...")
            config = SLOConfig(name="web_slo")
            slo = CognitiveSLO(config)
            logger.info("SLO initialized")
        
        # Process message through SLO
        result = slo.think(message, mode="auto")
        
        # Get knowledge stats
        knowledge = result.get('knowledge_stats', {}).get('total_documents', 0)
        
        return jsonify({
            'response': result.get('final_response', 'No response'),
            'confidence': result.get('confidence', 0),
            'systems_used': len(result.get('systems_used', [])),
            'knowledge': knowledge,
            'emotion': result.get('emotional_analysis', {}).get('emotion', 'neutral')
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learn', methods=['POST'])
def learn():
    global slo
    
    data = request.get_json()
    content = data.get('content', '')
    metadata = data.get('metadata', {})
    
    if not content:
        return jsonify({'error': 'No content provided'}), 400
    
    try:
        if slo is None:
            config = SLOConfig(name="web_slo")
            slo = CognitiveSLO(config)
        
        doc_id = slo.add_knowledge(content, metadata)
        
        return jsonify({
            'success': True,
            'doc_id': doc_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    if slo is None:
        return jsonify({
            'knowledge': 0,
            'sessions': 0,
            'systems': 16
        })
    
    return jsonify({
        'knowledge': slo.rag_engine.get_knowledge_stats().get('total_documents', 0),
        'sessions': len(slo.cognitive_arch.session_memory.conversation),
        'systems': 16
    })

def main():
    print("=" * 50)
    print("  SLO Web Interface")
    print("=" * 50)
    print("  Open: http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
