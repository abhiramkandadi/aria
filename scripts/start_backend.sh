#!/bin/bash
set -e
echo "Starting ARIA backend..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "ERROR: Ollama not running. Run: ollama serve"
    exit 1
fi
if ! ollama list | grep -q "qwen2.5"; then
    echo "Pulling Qwen 2.5 7B..."
    ollama pull qwen2.5:7b-instruct
fi
echo "Starting WebSocket bridge on ws://0.0.0.0:8765"
cd "$(dirname "$0")/../backend"
python bridge/server.py
