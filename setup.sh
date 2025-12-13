#!/bin/bash

# College Recommendation System - Setup Script
echo "🎓 Setting up College Recommendation System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    echo "📥 Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Setup Backend
echo "🚀 Setting up Backend..."
cd backend

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Install existing dependencies for the AI model
echo "📦 Installing AI model dependencies..."
pip3 install faiss-cpu sentence-transformers pandas scikit-learn

echo "✅ Backend setup complete!"

# Setup Frontend
echo "🎨 Setting up Frontend..."
cd ../frontend

# Install Node dependencies
echo "📦 Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete!"

echo ""
echo "🎉 Setup Complete! 🎉"
echo ""
echo "To start the system:"
echo "1. Start the backend:"
echo "   cd backend && python3 main.py"
echo ""
echo "2. In another terminal, start the frontend:"
echo "   cd frontend && npm start"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""