#!/usr/bin/env bash
set -e

echo "⚡ Starting People Lens Development Environment..."

# Trap SIGINT/SIGTERM to kill child processes
trap 'kill 0' EXIT

# Start Backend
echo "🚀 Launching FastAPI Backend on http://localhost:8000..."
source venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000 &

# Start Frontend
echo "✨ Launching Next.js 16 Frontend on http://localhost:3000..."
cd frontend
npm run dev &

wait
