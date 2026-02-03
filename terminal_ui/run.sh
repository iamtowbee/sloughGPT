#!/bin/bash

# TermShell - Terminal UI Dashboard
# Interactive terminal shell with dashboard, file browser, process monitor, and logs

echo "Starting TermShell..."
echo "Ensure you are running this in a real terminal (TTY), not in a pipe or redirected output."
echo ""

cd "$(dirname "$0")"
./target/release/terminal_ui

echo ""
echo "TermShell exited."