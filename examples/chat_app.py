#!/usr/bin/env python3
"""
SloughGPT Chat Application Example
Real-time chat interface with multiple models and conversation management
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatSession:
    """Chat session with conversation history"""
    
    def __init__(self, session_id: str, user_id: int = 1):
        self.session_id = session_id
        self.user_id = user_id
        self.messages = []
        self.model_name = "sloughgpt-base"
        self.temperature = 0.7
        self.max_tokens = 2048
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
    def get_context(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation context"""
        return self.messages[-limit:] if limit > 0 else self.messages
    
    def get_prompt(self) -> str:
        """Generate prompt from conversation history"""
        prompt_parts = []
        
        for msg in self.get_context():
            role_indicator = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            prompt_parts.append(f"{role_indicator}: {msg['content']}")
        
        prompt_parts.append("🤖 Assistant:")
        return "\n".join(prompt_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "message_count": len(self.messages)
        }

class ChatApplication:
    """Main chat application with SloughGPT integration"""
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.active_session: Optional[ChatSession] = None
        
    async def initialize(self):
        """Initialize chat application"""
        try:
            # Try to import SloughGPT components
            from sloughgpt.neural_network import SloughGPT
            from sloughgpt.config import ModelConfig
            from sloughgpt.cost_optimization import track_inference_cost
            
            # Load model (simplified for demo)
            config = ModelConfig()
            self.model = None  # Would be SloughGPT(config) in real implementation
            self.track_cost = track_inference_cost
            
            logger.info("✅ Chat application initialized successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️  Some dependencies not available: {e}")
            logger.info("Running in demo mode with mock responses")
            self.model = None
            self.track_cost = None
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {str(e)}")
            return False
    
    def create_session(self, session_id: str, user_id: int = 1) -> ChatSession:
        """Create a new chat session"""
        session = ChatSession(session_id, user_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get existing session"""
        return self.sessions.get(session_id)
    
    def switch_session(self, session_id: str) -> bool:
        """Switch to a different session"""
        if session_id in self.sessions:
            self.active_session = self.sessions[session_id]
            return True
        return False
    
    async def generate_response(self, session: ChatSession, prompt: str) -> Dict[str, Any]:
        """Generate response from the model"""
        # Add user message
        session.add_message("user", prompt)
        
        # Generate response
        if self.model:
            # Real model inference
            try:
                response_text = await self._real_inference(session)
                if self.track_cost:
                    # Track costs (mock data for demo)
                    input_tokens = len(prompt.split())
                    output_tokens = len(response_text.split())
                    self.track_cost(session.user_id, input_tokens, output_tokens, session.model_name)
            except Exception as e:
                response_text = f"❌ Model error: {str(e)}"
        else:
            # Demo mode with mock responses
            response_text = self._generate_mock_response(prompt)
        
        # Add assistant response
        session.add_message("assistant", response_text, {
            "model": session.model_name,
            "temperature": session.temperature,
            "tokens": len(response_text.split())
        })
        
        return {
            "response": response_text,
            "session_id": session.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model": session.model_name
        }
    
    async def _real_inference(self, session: ChatSession) -> str:
        """Perform real model inference"""
        # This would call the actual SloughGPT model
        # For now, return a placeholder
        return "This would be the actual SloughGPT response."
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for demo"""
        responses = [
            "I understand your question. Based on my training, I can help you with that.",
            "That's an interesting point! Let me provide some insights on that topic.",
            "I can help you with that! Here's what I think about your request.",
            "Thank you for asking! I'd be happy to assist you with this matter.",
            "That's a great question! Let me share what I know about this topic."
        ]
        
        # Simple keyword-based responses
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! I'm SloughGPT, your AI assistant. How can I help you today?"
        elif "bye" in prompt.lower() or "goodbye" in prompt.lower():
            return "Goodbye! It was great chatting with you. Feel free to come back anytime!"
        elif "help" in prompt.lower():
            return "I can help you with various tasks including answering questions, generating text, providing explanations, and more. What would you like help with?"
        else:
            # Return a semi-random response
            import random
            return random.choice(responses)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [
            {
                "session_id": session_id,
                "message_count": len(session.messages),
                "model": session.model_name,
                "last_activity": session.messages[-1]["timestamp"] if session.messages else None
            }
            for session_id, session in self.sessions.items()
        ]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.active_session and self.active_session.session_id == session_id:
                self.active_session = None
            return True
        return False

class ChatCLI:
    """Command-line interface for the chat application"""
    
    def __init__(self):
        self.app = ChatApplication()
        self.running = True
        
    async def start(self):
        """Start the CLI chat interface"""
        print("🤖 SloughGPT Chat Application")
        print("=" * 50)
        print("Type 'help' for commands, 'quit' to exit")
        print()
        
        # Initialize the application
        if not await self.app.initialize():
            print("❌ Failed to initialize chat application")
            return
        
        # Create default session
        default_session = self.app.create_session("default")
        self.app.active_session = default_session
        
        print(f"✅ Started new session: {default_session.session_id}")
        print()
        
        # Main chat loop
        while self.running:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Process commands
                if user_input.startswith('/'):
                    await self.process_command(user_input)
                else:
                    # Generate response
                    response = await self.app.generate_response(
                        self.app.active_session, user_input
                    )
                    print(f"🤖 Assistant: {response['response']}")
                    print()
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    async def process_command(self, command: str):
        """Process chat commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/help":
            self.show_help()
        elif cmd == "/quit" or cmd == "/exit":
            self.running = False
        elif cmd == "/new":
            session_id = parts[1] if len(parts) > 1 else f"session_{len(self.app.sessions)}"
            session = self.app.create_session(session_id)
            self.app.active_session = session
            print(f"✅ Created new session: {session_id}")
        elif cmd == "/switch":
            if len(parts) > 1:
                session_id = parts[1]
                if self.app.switch_session(session_id):
                    print(f"✅ Switched to session: {session_id}")
                else:
                    print(f"❌ Session not found: {session_id}")
            else:
                print("❌ Please specify a session ID")
        elif cmd == "/list":
            sessions = self.app.list_sessions()
            if sessions:
                print("📋 Active Sessions:")
                for session in sessions:
                    marker = "👉" if session["session_id"] == self.app.active_session.session_id else "  "
                    print(f"  {marker} {session['session_id']} ({session['message_count']} messages)")
            else:
                print("📭 No active sessions")
        elif cmd == "/delete":
            if len(parts) > 1:
                session_id = parts[1]
                if self.app.delete_session(session_id):
                    print(f"✅ Deleted session: {session_id}")
                else:
                    print(f"❌ Session not found: {session_id}")
            else:
                print("❌ Please specify a session ID")
        elif cmd == "/info":
            if self.app.active_session:
                session_info = self.app.active_session.to_dict()
                print(f"📊 Current Session Info:")
                print(f"   Session ID: {session_info['session_id']}")
                print(f"   Messages: {session_info['message_count']}")
                print(f"   Model: {session_info['model_name']}")
                print(f"   Temperature: {session_info['temperature']}")
                print(f"   Max Tokens: {session_info['max_tokens']}")
            else:
                print("❌ No active session")
        else:
            print(f"❌ Unknown command: {cmd}")
            print("Type '/help' for available commands")
    
    def show_help(self):
        """Show help information"""
        print("📚 Available Commands:")
        print("   /help          - Show this help message")
        print("   /quit, /exit   - Exit the chat application")
        print("   /new [id]     - Create a new session")
        print("   /switch <id>   - Switch to a different session")
        print("   /list          - List all active sessions")
        print("   /delete <id>   - Delete a session")
        print("   /info          - Show current session info")
        print()
        print("💡 Tips:")
        print("   • Type normally to chat with the AI")
        print("   • Commands start with '/'")
        print("   • Use multiple sessions for different conversations")

async def main():
    """Main function for the chat application"""
    parser = argparse.ArgumentParser(description="SloughGPT Chat Application")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--session", help="Start with specific session ID")
    
    args = parser.parse_args()
    
    # Start CLI
    cli = ChatCLI()
    await cli.start()

if __name__ == "__main__":
    asyncio.run(main())