#!/bin/bash
# Launch the Gradio Web UI

echo "=================================================="
echo "  🚀 Starting Agentic RAG Web Interface"
echo "=================================================="
echo ""
echo "Initializing system..."
echo ""

# Check if running in a container/codespace
if [ -n "$CODESPACE_NAME" ]; then
    echo "✓ Running in GitHub Codespaces"
    echo "✓ Your app will be available via the forwarded port"
    echo ""
fi

# Launch the Gradio app
python app.py
