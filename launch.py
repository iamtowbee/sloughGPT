#!/usr/bin/env python3
"""
SloughGPT Launcher

Simple launcher for the domain-based architecture.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def main():
    print("🚀 Starting SloughGPT Domain Architecture")
    print("=" * 50)
    
    # Test imports
    from domains.training import DatasetCreator, NanoGPT
    print("✓ Training domain")
    
    from domains.cognitive import CognitiveCore, KnowledgeGraph
    print("✓ Cognitive domain")
    
    from domains.infrastructure import HaulsStore
    print("✓ Infrastructure domain")
    
    from domains.ui import CLIInterface
    print("✓ UI domain")
    
    from domains.enterprise import AuthenticationService
    print("✓ Enterprise domain")
    
    print()
    print("🎉 All domains loaded successfully!")
    print()
    print("Available:")
    print("  from domains.training import DatasetCreator, NanoGPT")
    print("  from domains.cognitive import CognitiveCore, KnowledgeGraph")
    print("  from domains.infrastructure import HaulsStore")
    print("  from domains.ui import CLIInterface")


if __name__ == "__main__":
    asyncio.run(main())
