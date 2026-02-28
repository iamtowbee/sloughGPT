#!/usr/bin/env python3
"""
SloughGPT AI CLI - Personality-Aware Command Line Interface

A real AI assistant CLI with:
- Real computational personality metrics
- Multiple personalities
- Actual response generation (not mock)
- Conversation analysis
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from domains.ai_agent import PersonalityAwareAgent
from domains.ai_personality import AIPurpose


class AICLI:
    """Personality-aware AI CLI"""
    
    def __init__(self, personality: str = "chat", vocab_size: int = 1000):
        print(f"Loading SloughGPT ({personality})...", end=" ")
        self.agent = PersonalityAwareAgent(personality_name=personality, vocab_size=vocab_size)
        info = self.agent.get_personality_info()
        print(f"Ready!")
        print(f"  Personality: {info['name']} ({info['purpose']})")
        print(f"  Tone: {info['tone']}")
    
    def run(self, prompt: str):
        """Run single prompt"""
        result = self.agent.chat(prompt)
        
        print(f"\n🤖 {result['response']}")
        print(f"\n📊 Metrics:")
        print(f"   Friendliness:  {result['metrics']['friendliness']:.3f}")
        print(f"   Helpfulness:  {result['metrics']['helpfulness']:.3f}")
        print(f"   Creativity:    {result['metrics']['creativity']:.3f}")
        print(f"   Formality:    {result['metrics']['formality']:.3f}")
        
        return result
    
    def repl(self):
        """Interactive REPL"""
        print("\n" + "="*50)
        print("  SloughGPT AI - Interactive Mode")
        print("="*50)
        print("Commands:")
        print("  /personality - Show current personality")
        print("  /switch <name> - Switch personality (chat, coder, writer, teacher)")
        print("  /metrics - Show conversation metrics")
        print("  /help - Show this help")
        print("  /exit - Exit")
        print("="*50 + "\n")
        
        while True:
            try:
                prompt = input("\n👤 ").strip()
                
                if not prompt:
                    continue
                
                if prompt.startswith("/"):
                    self._handle_command(prompt)
                    continue
                
                if prompt.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break
                
                result = self.agent.chat(prompt)
                print(f"\n🤖 {result['response']}")
                print(f"   📊 F:{result['metrics']['friendliness']:.2f} "
                      f"H:{result['metrics']['helpfulness']:.2f} "
                      f"C:{result['metrics']['creativity']:.2f}")
                
            except KeyboardInterrupt:
                print("\n(Use /exit to quit)")
            except EOFError:
                break
    
    def _handle_command(self, cmd: str):
        """Handle slash commands"""
        cmd = cmd.lower().strip()
        
        if cmd == "/help":
            print("""
Commands:
  /personality     - Show current personality info
  /switch <name>   - Switch personality (chat, coder, writer, teacher)
  /metrics         - Show conversation metrics
  /help            - Show this help
  /exit            - Exit
            """)
        
        elif cmd == "/personality":
            info = self.agent.get_personality_info()
            print(f"""
Current Personality:
  Name: {info['name']}
  Purpose: {info['purpose']}
  Description: {info['description']}
  Tone: {info['tone']}
  Creativity: {info['creativity']}
  Domains: {info['domains']}
            """)
        
        elif cmd.startswith("/switch"):
            parts = cmd.split()
            if len(parts) > 1:
                name = parts[1]
                if self.agent.switch_personality(name):
                    info = self.agent.get_personality_info()
                    print(f"Switched to: {info['name']} ({info['purpose']})")
                else:
                    print(f"Unknown personality: {name}")
                    print("Available: chat, coder, writer, teacher")
            else:
                print("Usage: /switch <name>")
        
        elif cmd == "/metrics":
            analysis = self.agent.analyze_conversation()
            print(f"""
Conversation Analysis:
  Messages: {analysis['message_count']}
  
  Averages:
    Friendliness:  {analysis['averages']['friendliness']:.3f}
    Helpfulness:   {analysis['averages']['helpfulness']:.3f}
    Creativity:    {analysis['averages']['creativity']:.3f}
    Formality:     {analysis['averages']['formality']:.3f}
            """)
        
        elif cmd == "/exit":
            print("\nGoodbye!")
            sys.exit(0)
        
        else:
            print(f"Unknown command: {cmd}")


def main():
    parser = argparse.ArgumentParser(description="SloughGPT AI CLI")
    parser.add_argument("prompt", nargs="?", help="Prompt to send")
    parser.add_argument("-p", "--personality", default="chat",
                       choices=["chat", "coder", "writer", "teacher"],
                       help="AI personality")
    parser.add_argument("-v", "--vocab-size", type=int, default=1000,
                       help="Vocabulary size for model")
    parser.add_argument("-i", "--interactive", action="store_true",
                       help="Interactive REPL mode")
    args = parser.parse_args()
    
    cli = AICLI(personality=args.personality, vocab_size=args.vocab_size)
    
    if args.prompt:
        cli.run(args.prompt)
    elif args.interactive:
        cli.repl()
    else:
        cli.repl()


if __name__ == "__main__":
    main()
