"""
UI Domain - Recovered Files Reference (legacy filenames → current monorepo locations).
"""

RECOVERED_FILES = {
    "sloughgpt.py": {"description": "CLI entry (historical name)", "location": "cli.py"},
    "sloughgpt_integrated.py": {
        "description": "Integrated variant (not shipped)",
        "location": "(legacy — use apps/ + domains/)",
    },
    "web_interface.py": {"description": "Web UI", "location": "apps/web/web/"},
    "simple_webui.py": {"description": "Simple WebUI", "location": "apps/web/web/"},
    "enhanced_webui.py": {"description": "Enhanced WebUI", "location": "apps/web/web/"},
    "api_server.py": {"description": "REST API server", "location": "packages/core-py/domains/ui/api_server.py"},
    "simple_api_server.py": {
        "description": "Thin API wrapper",
        "location": "apps/api/server/main.py",
    },
    "chat.py": {"description": "Chat UI routes", "location": "apps/web/web/app/(app)/chat/"},
    "build_webui.py": {"description": "WebUI build", "location": "apps/web/web/package.json (npm run build)"},
    "launch_webui.py": {"description": "WebUI dev server", "location": "apps/web/web (npm run dev)"},
}

__all__ = ["RECOVERED_FILES"]
