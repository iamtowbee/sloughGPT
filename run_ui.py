#!/usr/bin/env python3
"""
Launch OpenWebUI with SLO integration.

Uses the production OpenWebUI codebase with SLO model integration.
"""

import os
import sys
from pathlib import Path

# Set repo root for imports
ROOT = Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if __name__ == "__main__":
    print("🚀 Starting OpenWebUI with SLO integration...")
    print("📱 Web interface will be available at: http://localhost:8080")
    print("📝 SLO model will be available in the model selection dropdown")
    
    # Execute OpenWebUI with SLO integration (our ui_slo_integration module is auto-imported)
    exec(open("openwebui/upstream/backend/open_webui/main.py").read())