"""
CLI Shortcuts - Ported from recovered cli_shortcuts.py
"""

from pathlib import Path
from typing import Dict, Optional


class CLIManager:
    """Manage CLI aliases and shortcuts."""
    
    def __init__(self, aliases_file: str = ".slogpt_aliases"):
        self.aliases_file = Path(aliases_file)
        self.shortcuts = self._define_shortcuts()
        self.aliases = self._define_aliases()
    
    def _define_shortcuts(self) -> Dict[str, Dict]:
        """Define common shortcuts."""
        return {
            "new": {
                "command": "create_dataset",
                "description": "Create new dataset",
                "usage": "slo new mydata 'your text here'"
            },
            "train": {
                "command": "train",
                "description": "Train on dataset",
                "usage": "slo train mydata"
            },
            "list": {
                "command": "list",
                "description": "List available datasets",
                "usage": "slo list"
            },
            "mix": {
                "command": "mix",
                "description": "Create mixed dataset",
                "usage": "slo mix web:0.7,code:0.3 mixed_data"
            },
            "validate": {
                "command": "validate",
                "description": "Validate dataset quality",
                "usage": "slo validate mydata"
            },
        }
    
    def _define_aliases(self) -> Dict[str, str]:
        """Define command aliases."""
        return {
            "l": "list",
            "t": "train",
            "n": "new",
            "v": "validate",
            "m": "mix",
        }
    
    def get_command(self, shortcut: str) -> Optional[str]:
        """Get command for a shortcut."""
        if shortcut in self.aliases:
            shortcut = self.aliases[shortcut]
        
        if shortcut in self.shortcuts:
            return self.shortcuts[shortcut]["command"]
        return None
    
    def show_help(self, shortcut: Optional[str] = None) -> str:
        """Show help for shortcuts."""
        if shortcut:
            if shortcut in self.shortcuts:
                s = self.shortcuts[shortcut]
                return f"{shortcut}: {s['description']}\nUsage: {s['usage']}"
            return f"Unknown shortcut: {shortcut}"
        
        lines = ["Available shortcuts:"]
        for name, s in self.shortcuts.items():
            lines.append(f"  {name:10} - {s['description']}")
        return "\n".join(lines)


__all__ = ["CLIManager"]
