#!/usr/bin/env python3
"""
CLI Aliases and Shortcuts for SloGPT Dataset System

Provides convenient aliases for common operations.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional


class CLIManager:
    """Manage CLI aliases and shortcuts."""
    
    def __init__(self):
        self.aliases_file = Path(".slogpt_aliases")
        self.shortcuts = self._define_shortcuts()
        self.aliases = self._define_aliases()
    
    def _define_shortcuts(self) -> Dict[str, Dict]:
        """Define common shortcuts."""
        return {
            "new": {
                "command": "create_dataset_fixed.py",
                "description": "Create new dataset",
                "usage": "slo new mydata 'your text here'"
            },
            "train": {
                "command": "train_simple.py",
                "description": "Train on dataset",
                "usage": "slo train mydata"
            },
            "list": {
                "command": "train_simple.py --list",
                "description": "List available datasets",
                "usage": "slo list"
            },
            "mix": {
                "command": "dataset_manager.py mix",
                "description": "Create mixed dataset",
                "usage": "slo mix web:0.7,code:0.3 mixed_data"
            },
            "validate": {
                "command": "advanced_dataset_features.py validate",
                "description": "Validate dataset quality",
                "usage": "slo validate mydata"
            },
            "version": {
                "command": "advanced_dataset_features.py version",
                "description": "Create dataset version",
                "usage": "slo version mydata --tag v1.0.0"
            },
            "restore": {
                "command": "advanced_dataset_features.py restore",
                "description": "Restore dataset version",
                "usage": "slo restore mydata v1.0.0"
            },
            "prepare": {
                "command": "universal_prepare.py",
                "description": "Prepare dataset from source",
                "usage": "slo prepare mydata --source ./data_folder"
            },
            "monitor": {
                "command": "performance_optimizer.py monitor",
                "description": "Monitor training performance",
                "usage": "slo monitor"
            },
            "optimize": {
                "command": "performance_optimizer.py optimize",
                "description": "Get optimized training command",
                "usage": "slo optimize mydata"
            },
            "batch": {
                "command": "batch_processor.py",
                "description": "Process multiple datasets",
                "usage": "slo batch --config batch_config.yaml"
            },
            "status": {
                "command": "advanced_dataset_features.py validate",
                "description": "Show dataset status",
                "usage": "slo status mydata"
            }
        }
    
    def _define_aliases(self) -> Dict[str, str]:
        """Define command aliases."""
        return {
            "slo": "python3 slo_cli.py",
            "sl": "python3 slo_cli.py",
            "slogpt": "python3 slo_cli.py"
        }
    
    def install_aliases(self) -> bool:
        """Install CLI aliases to shell."""
        print("🔧 Installing SloGPT CLI aliases...")
        
        # Create slo_cli.py main wrapper
        cli_script = self._create_cli_wrapper()
        with open("slo_cli.py", "w") as f:
            f.write(cli_script)
        
        # Make executable
        os.chmod("slo_cli.py", 0o755)
        
        # Generate shell alias commands
        alias_commands = []
        for alias, command in self.aliases.items():
            alias_commands.append(f"alias {alias}='{command}'")
        
        # Save aliases to file
        with open(".slogpt_aliases", "w") as f:
            f.write("\n".join(alias_commands))
        
        print("✅ CLI aliases created!")
        print("\n📝 Add to your shell (.bashrc, .zshrc, etc.):")
        print(f"source {Path.cwd()}/.slogpt_aliases")
        
        return True
    
    def _create_cli_wrapper(self) -> str:
        """Create main CLI wrapper script."""
        return '''#!/usr/bin/env python3
"""
SloGPT CLI Wrapper - Main Entry Point

Quick access to all SloGPT dataset features.
"""

import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("🚀 SloGPT Dataset System")
        print("=" * 40)
        print("Available shortcuts:")
        print("  slo new <name> <text>     - Create dataset")
        print("  slo train <name>           - Train model") 
        print("  slo list                   - List datasets")
        print("  slo mix <ratios> <output>  - Mix datasets")
        print("  slo validate <name>       - Validate dataset")
        print("  slo version <name> <tag>  - Create version")
        print("  slo restore <name> <ver>   - Restore version")
        print("  slo prepare <name> <src>  - Prepare dataset")
        print("  slo monitor               - Monitor performance")
        print("  slo optimize <name>       - Optimize training")
        print("  slo batch <config>         - Batch process")
        print("  slo status <name>         - Show status")
        print("\\nExamples:")
        print("  slo new mydata 'training text'")
        print("  slo train mydata")
        print("  slo mix web:0.7,code:0.3 mixed")
        print("  slo validate mydata")
        return
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    # Command mappings
    shortcuts = {
        "new": ["python3", "create_dataset_fixed.py"],
        "train": ["python3", "train_simple.py"],
        "list": ["python3", "train_simple.py", "--list"],
        "validate": ["python3", "advanced_dataset_features.py", "validate"],
        "version": ["python3", "advanced_dataset_features.py", "version"],
        "restore": ["python3", "advanced_dataset_features.py", "restore"],
        "prepare": ["python3", "universal_prepare.py"],
        "monitor": ["python3", "performance_optimizer.py", "monitor"],
        "optimize": ["python3", "performance_optimizer.py", "optimize"],
        "batch": ["python3", "batch_processor.py"],
        "status": ["python3", "advanced_dataset_features.py", "validate"]
    }
    
    # Special handling for mix command
    if command == "mix":
        if len(args) >= 2:
            ratios, output = args[0], args[1]
            cmd = ["python3", "dataset_manager.py", "mix", "--ratios", ratios, "--output", output]
        else:
            print("❌ Usage: slo mix <ratios> <output>")
            print("Example: slo mix web:0.7,code:0.3 mixed_data")
            return
    else:
        base_cmd = shortcuts.get(command)
        if not base_cmd:
            print(f"❌ Unknown command: {command}")
            print("Run 'slo' to see available commands")
            return
        
        cmd = base_cmd + args
    
    # Execute command
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Command completed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\\n⏹ Command interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    def show_help(self):
        """Show help information."""
        print("🚀 SloGPT CLI Shortcuts & Aliases")
        print("=" * 50)
        
        print("\n📝 Available Shortcuts:")
        for name, info in self.shortcuts.items():
            print(f"  slo {name:<12} - {info['description']}")
            print(f"{' ' * 15}{info['usage']}")
        
        print("\n🔧 Shell Aliases:")
        for alias, command in self.aliases.items():
            print(f"  {alias:<8} -> {command}")
        
        print("\n💡 Installation:")
        print("  python3 cli_shortcuts.py --install")
        print("  Then add: source $(pwd)/.slogpt_aliases to your shell config")
    
    def run_command(self, command: str, args: Optional[List[str]] = None):
        """Run a command through the CLI system."""
        if args is None:
            args = []
        
        shortcut_info = self.shortcuts.get(command)
        if not shortcut_info:
            print(f"❌ Unknown shortcut: {command}")
            self.show_help()
            return False
        
        base_cmd = shortcut_info["command"]
        
        # Special handling for complex commands
        if command == "mix":
            if len(args) < 2:
                print("❌ Mix requires ratios and output name")
                print(f"Usage: {shortcut_info['usage']}")
                return False
            ratios, output = args[0], args[1]
            cmd = ["python3", base_cmd, "--ratios", ratios, "--output", output]
        else:
            cmd = ["python3", base_cmd] + args
        
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            return False
    
    def create_quick_scripts(self):
        """Create quick access scripts."""
        print("📜 Creating quick access scripts...")
        
        scripts = {
            "quick_train.py": '''#!/usr/bin/env python3
"""Quick training script."""
import subprocess
import sys

dataset = sys.argv[1] if len(sys.argv) > 1 else "default"
cmd = f"python3 train_simple.py {dataset}"
subprocess.run(cmd, shell=True)
''',
            "quick_new.py": '''#!/usr/bin/env python3
"""Quick dataset creation."""
import subprocess
import sys

if len(sys.argv) < 3:
    print("Usage: python3 quick_new.py <name> <text>")
    sys.exit(1)

name, text = sys.argv[1], sys.argv[2]
cmd = f'python3 create_dataset_fixed.py {name} "{text}"'
subprocess.run(cmd, shell=True)
''',
            "quick_list.py": '''#!/usr/bin/env python3
"""Quick dataset listing."""
import subprocess
subprocess.run("python3 train_simple.py --list", shell=True)
'''
        }
        
        for script_name, content in scripts.items():
            with open(script_name, "w") as f:
                f.write(content)
            os.chmod(script_name, 0o755)
        
        print("✅ Quick scripts created:")
        for script in scripts.keys():
            print(f"  📜 {script}")
    
    def setup_environment(self):
        """Setup complete environment with all features."""
        print("🏗️ Setting up complete SloGPT environment...")
        
        # Install aliases
        self.install_aliases()
        
        # Create quick scripts
        self.create_quick_scripts()
        
        # Create workspace structure
        workspace_dirs = [
            "datasets",
            "configs", 
            "models",
            "logs",
            "exports",
            "backups"
        ]
        
        for dir_name in workspace_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        print("✅ Environment setup complete!")
        print("\n📁 Created workspace structure:")
        for dir_name in workspace_dirs:
            print(f"  📁 {dir_name}/")
        
        print("\n🚀 You're ready to use SloGPT!")
        print("Start with: slo")


def main():
    """Main CLI for shortcuts and aliases."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloGPT CLI shortcuts and aliases")
    parser.add_argument('--install', action='store_true', help='Install shell aliases')
    parser.add_argument('--help-shortcuts', action='store_true', help='Show available shortcuts')
    parser.add_argument('--setup', action='store_true', help='Setup complete environment')
    parser.add_argument('--create-scripts', action='store_true', help='Create quick access scripts')
    
    args, unknown = parser.parse_known_args()
    
    cli_manager = CLIManager()
    
    if args.install:
        cli_manager.install_aliases()
    elif args.help_shortcuts:
        cli_manager.show_help()
    elif args.setup:
        cli_manager.setup_environment()
    elif args.create_scripts:
        cli_manager.create_quick_scripts()
    else:
        cli_manager.show_help()


if __name__ == "__main__":
    main()