#!/usr/bin/env python3
"""
SloughGPT CLI - Command Line Interface

A Claude Code / OpenCode-style CLI that uses the ML wrapper for inference.

Usage:
    python sloughgpt_cli.py                    # Interactive REPL
    python sloughgpt_cli.py "your prompt"      # Single query
    python sloughgpt_cli.py --file prompt.txt # From file
    python sloughgpt_cli.py --help            # Help
"""

import sys
import os
import argparse
import readline
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wrapper import SloughGPTWrapper


class SloughGPTCLI:
    """SloughGPT Command Line Interface"""
    
    def __init__(self, model_type="gpt", vocab_size=50000, max_length=512):
        print("Loading SloughGPT...", end=" ")
        self.wrapper = SloughGPTWrapper(
            model_type=model_type,
            vocab_size=vocab_size,
            max_length=max_length
        )
        print(f"Ready! (v{self.wrapper.get_version()})")
        
        self.history = []
        self.session_id = 0
    
    def chat(self, message: str, stream: bool = True) -> str:
        """Send a message and get response"""
        result = self.wrapper.run(message)
        
        self.history.append({
            "role": "user",
            "content": message
        })
        self.history.append({
            "role": "assistant", 
            "content": result.get("output", "")
        })
        
        return result.get("output", "")
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from prompt"""
        return self.wrapper.generate(prompt, max_new_tokens=max_tokens)
    
    def batch(self, prompts: list) -> list:
        """Process multiple prompts"""
        return self.wrapper.batch(prompts)
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
        print("History cleared.")
    
    def show_history(self, limit: int = 10):
        """Show conversation history"""
        if not self.history:
            print("No history.")
            return
        
        print(f"\n--- Conversation History (last {limit}) ---\n")
        for i, msg in enumerate(self.history[-limit:]):
            role = msg["role"].upper()
            content = msg["content"]
            if role == "USER":
                print(f"\n❯ {content}")
            else:
                print(f"\n➜ {content[:200]}{'...' if len(content) > 200 else ''}")
        print()
    
    def repl(self):
        """Interactive REPL mode"""
        print("\n" + "="*50)
        print("  SloughGPT CLI - Interactive Mode")
        print("="*50)
        print("Commands:")
        print("  /clear   - Clear history")
        print("  /history - Show history")
        print("  /help    - Show this help")
        print("  /exit    - Exit")
        print("="*50 + "\n")
        
        while True:
            try:
                prompt = input("\n❯ ").strip()
                
                if not prompt:
                    continue
                
                if prompt.startswith("/"):
                    self.handle_command(prompt)
                    continue
                
                if prompt.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break
                
                response = self.chat(prompt)
                print(f"\n➜ {response}")
                
            except KeyboardInterrupt:
                print("\n\nUse /exit to quit.")
            except EOFError:
                break
    
    def handle_command(self, cmd: str):
        """Handle slash commands"""
        cmd = cmd.lower().strip()
        
        if cmd == "/clear":
            self.clear_history()
        elif cmd == "/history":
            self.show_history()
        elif cmd == "/help":
            print("""
Commands:
  /clear      - Clear conversation history
  /history    - Show conversation history  
  /help       - Show this help
  /exit       - Exit the CLI

Tips:
  - Use ↑/↓ arrows for command history
  - Multi-line input supported
  - Pipe input from files: cat prompt.txt | python sloughgpt_cli.py
            """)
        elif cmd == "/exit":
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print(f"Unknown command: {cmd}")


def main():
    parser = argparse.ArgumentParser(
        description="SloughGPT CLI - AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello, how are you?"
  %(prog)s --file prompt.txt
  %(prog)s --generate "Once upon a time" --max-tokens 100
  %(prog)s --interactive
        """
    )
    
    parser.add_argument("prompt", nargs="?", default="", help="Prompt to send")
    parser.add_argument("-f", "--file", help="Read prompt from file")
    parser.add_argument("-g", "--generate", help="Generate text (like OpenAI)")
    parser.add_argument("-m", "--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("-v", "--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--model", default="gpt", help="Model type")
    
    args = parser.parse_args()
    
    cli = SloughGPTCLI(
        model_type=args.model,
        vocab_size=args.vocab_size,
        max_length=args.max_length
    )
    
    prompt = ""
    
    if args.file:
        with open(args.file) as f:
            prompt = f.read()
    elif args.generate:
        result = cli.generate(args.generate, args.max_tokens)
        print(result)
        return
    elif args.prompt:
        prompt = args.prompt
    elif args.interactive:
        cli.repl()
        return
    else:
        cli.repl()
        return
    
    if prompt:
        result = cli.chat(prompt)
        print(result)


if __name__ == "__main__":
    main()
