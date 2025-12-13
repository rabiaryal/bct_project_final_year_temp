# 🎓 College Recommendation Chatbot - Setup Guide

A comprehensive guide to set up and run the College Recommendation System with XGBoost-based recommendations.

---

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher
- **Git**: For cloning the repository

---

## 🚀 Step-by-Step Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd bct_final_year_project
```

---

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when activated.

---

### Step 3: Install Python Dependencies

```bash
# Install backend requirements
pip install -r backend/requirements.txt

# Install additional ML dependencies
pip install torch transformers sentence-transformers faiss-cpu xgboost pandas numpy scikit-learn
```

**Full list of required packages:**
- `fastapi` - Web framework for the API
- `uvicorn` - ASGI server
- `torch` - PyTorch for ML models
- `transformers` - Hugging Face transformers
- `sentence-transformers` - For embeddings
- `faiss-cpu` - Vector similarity search
- `xgboost` - XGBoost recommendation model
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - ML utilities

---

### Step 4: Verify Data Files

Ensure the following files exist in the project root:

```
bct_final_year_project/
├── full_data.json          # College database (required)
├── data.json               # Alternative data file
└── models/                 # Pre-trained models folder
    ├── primary_intent_model/
    ├── sub_intent_model/
    ├── entity_model/
    ├── faiss_index/
    └── xgboost_recommender/
```

> **Note**: The XGBoost model will be automatically trained on first run if not present.

---

### Step 5: Start the Backend Server

Open a terminal and run:

```bash
# Navigate to backend folder
cd backend

# Start the FastAPI server
python main.py
```

**Expected output:**
```
🏗️ Initializing Unified Pipeline Controller...
🔥 Hierarchical Intent Detection enabled
🏷️  Transformer Entity Extraction enabled
🎯 Structured Retriever enabled
⚡ FAISS Direct Answer System enabled
🌲 XGBoost Recommendation System enabled
✅ Unified Pipeline Controller ready
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The backend will be available at: `http://localhost:8000`

---

### Step 6: Install Frontend Dependencies

Open a **new terminal** and run:

```bash
# Navigate to frontend folder
cd frontend

# Install npm packages
npm install
```

---

### Step 7: Start the Frontend

```bash
# Start React development server
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view college-recommendation-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

The frontend will automatically open in your browser at: `http://localhost:3000`

---

## ✅ Verification

### Test the System

1. Open your browser to `http://localhost:3000`
2. Try these sample queries:
   - `"Hello"` - Test greeting
   - `"Recommend colleges for civil engineering in Kathmandu"` - Recommendation
   - `"What is the fee at Pulchowk?"` - Direct question
   - `"Compare KU and Thapathali"` - Comparison

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/chat` | POST | Send message to chatbot |
| `/ws/{session_id}` | WebSocket | Real-time chat connection |

---

## 🔧 Troubleshooting

### Common Issues

**1. Port already in use**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

**2. Module not found errors**
```bash
# Reinstall dependencies
pip install --upgrade -r backend/requirements.txt
pip install torch transformers sentence-transformers faiss-cpu xgboost
```

**3. CUDA/GPU issues**
```bash
# Use CPU-only versions
pip install faiss-cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**4. Frontend can't connect to backend**
- Ensure backend is running on port 8000
- Check CORS settings in `backend/main.py`
- Verify WebSocket connection URL in frontend

---

## 📁 Project Structure

```
bct_final_year_project/
├── backend/
│   ├── main.py              # FastAPI server
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   └── index.js         # Entry point
│   └── package.json         # npm dependencies
├── app/
│   └── unified_pipeline.py  # Main chatbot pipeline
├── core/
│   ├── intent/              # Intent detection
│   ├── qa/                  # Question answering
│   └── recommendation/      # XGBoost recommender
├── models/                  # Pre-trained models
├── full_data.json          # College database
└── how_to_run.md           # This file
```

---

## 🛑 Stopping the System

1. **Stop Frontend**: Press `Ctrl+C` in the frontend terminal
2. **Stop Backend**: Press `Ctrl+C` in the backend terminal
3. **Deactivate Virtual Environment**: Run `deactivate`

---

## 📞 Support

If you encounter any issues:
1. Check the terminal logs for error messages
2. Verify all dependencies are installed
3. Ensure data files are present
4. Check that required ports (3000, 8000) are available

---

**Happy coding! 🚀**
