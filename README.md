# College Recommendation Dialogue System

An AI-powered chatbot for Nepal engineering college information and recommendations. Built with **FastAPI** (backend), **React** (frontend), **MongoDB Atlas** (database), and custom-trained **BERT** + **RoBERTa+CRF** NLU models.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Clone the Repository](#step-1--clone-the-repository)
4. [Step 2 — Set Up the Python Environment](#step-2--set-up-the-python-environment)
5. [Step 3 — Set Up MongoDB Atlas](#step-3--set-up-mongodb-atlas)
6. [Step 4 — Configure Environment Variables](#step-4--configure-environment-variables)
7. [Step 5 — Train the Models](#step-5--train-the-models)
8. [Step 6 — Run the Backend](#step-6--run-the-backend)
9. [Step 7 — Run the Frontend](#step-7--run-the-frontend)
10. [API Endpoints](#api-endpoints)
11. [Architecture Overview](#architecture-overview)

---

## Project Structure

```
bct_final_year_project/
├── backend/
│   ├── app/
│   │   ├── api/                  # FastAPI routes (chat, health)
│   │   ├── context/              # Dialogue state tracking (SlotManager)
│   │   ├── core/                 # Query builder, scorer, slot filler
│   │   ├── handlers/             # Intent-specific handlers + router
│   │   ├── nlu/
│   │   │   ├── intent/
│   │   │   │   ├── bert_intent.py          # BERT intent classifier (runtime)
│   │   │   │   └── train_bert_intent.py    # BERT training script
│   │   │   └── entity/
│   │   │       ├── roberta_ner.py          # RoBERTa+CRF NER (runtime)
│   │   │       ├── train_with_crf.py       # CRF entity training script
│   │   │       └── data_loader.py          # NER data loader
│   │   ├── repositories/         # MongoDB data access
│   │   ├── templates/            # Response templates & formatters
│   │   ├── utils/                # Config, logger, constants
│   │   ├── dialogue_manager.py   # Main orchestrator
│   │   └── main.py               # FastAPI entry point
│   └── requirements.txt
│
├── data/
│   ├── intent_entity.json        # Intent training data (text + intent label)
│   ├── bio_dataset.json          # Entity training data (BIO-tagged tokens)
│   ├── full_data.json            # College data (reference)
│   └── colleges_lower.json       # Lowercase college data (reference)
│
├── models/                       # ⚠️ NOT in repo — you must train these
│   ├── bert_intent_model/        # Trained BERT intent classifier
│   └── crf_entity_model/         # Trained RoBERTa+CRF entity model
│
├── frontend/                     # React frontend
│   ├── public/
│   └── src/
│
├── sql 123.json                  # MongoDB seed data (15 colleges)
├── .env.mongodb                  # MongoDB connection config
└── docs/                         # Documentation & training graphs
```

---

## Prerequisites

| Tool          | Version   | Notes                                     |
| ------------- | --------- | ----------------------------------------- |
| **Python**    | 3.11+     | Required for PyTorch + Transformers       |
| **Conda**     | any       | For environment management                |
| **Node.js**   | 18+       | For the React frontend                    |
| **npm**       | 9+        | Comes with Node.js                        |
| **MongoDB Atlas** | Free tier | Cloud database (no local install needed) |
| **Git**       | any       | To clone the repo                         |

> **GPU (optional):** Training is faster on a CUDA/MPS GPU but works on CPU too.

---

## Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd bct_final_year_project
```

---

## Step 2 — Set Up the Python Environment

### 2.1 Create a Conda environment

```bash
conda create -n bctproject python=3.11 -y
conda activate bctproject
```

### 2.2 Install runtime dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2.3 Install training dependencies

These are only needed if you want to train models (Step 5). They are **not** needed to just run the server.

```bash
pip install scikit-learn pandas numpy matplotlib seaborn seqeval
```

> `requirements.txt` already includes `torch`, `transformers`, `pytorch-crf`, `rapidfuzz`, and all runtime deps.

---

## Step 3 — Set Up MongoDB Atlas

The chatbot uses **MongoDB Atlas** as its database. The file `sql 123.json` in the project root contains all 15 colleges that need to be imported.

### 3.1 Create a MongoDB Atlas account

1. Go to [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas) and sign up (free tier is fine).
2. Create a new **Cluster** (the free M0 tier works).
3. Wait for the cluster to be provisioned.

### 3.2 Create a database user

1. In the Atlas dashboard, go to **Database Access** (left sidebar).
2. Click **Add New Database User**.
3. Choose **Password** authentication.
4. Set a **username** and **password** (remember these — you need them for the connection string).
5. Set privileges to **Read and write to any database**.
6. Click **Add User**.

### 3.3 Allow network access

1. Go to **Network Access** (left sidebar).
2. Click **Add IP Address**.
3. Click **Allow Access from Anywhere** (adds `0.0.0.0/0`) — or add your specific IP.
4. Click **Confirm**.

### 3.4 Get the connection string

1. Go to **Database** (left sidebar) → click **Connect** on your cluster.
2. Choose **Drivers** (Connect your application).
3. Copy the connection string. It looks like:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
4. Replace `<username>` and `<password>` with the credentials from Step 3.2.

### 3.5 Import the college data

You need to create a database called `crs` with a collection called `college data`, then import the 15 college documents from `sql 123.json`.

**Option A — Using MongoDB Compass (GUI, easiest):**

1. Download [MongoDB Compass](https://www.mongodb.com/products/compass) (free).
2. Paste your connection string and click **Connect**.
3. Click **Create Database**:
   - Database Name: `crs`
   - Collection Name: `college data`
4. Open the `college data` collection → click **Add Data** → **Import JSON**.
5. Select the file `sql 123.json` from the project root.
6. Click **Import** — you should see 15 documents.

**Option B — Using `mongoimport` (CLI):**

```bash
# Install MongoDB Database Tools first:
# macOS: brew install mongodb-database-tools
# Ubuntu: sudo apt install mongodb-database-tools
# Windows: download from https://www.mongodb.com/try/download/database-tools

# Then import (replace the connection string with yours):
mongoimport \
  --uri "mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/crs" \
  --collection "college data" \
  --file "sql 123.json" \
  --jsonArray
```

**Option C — Using a Python script:**

```bash
# From the project root:
cd backend
conda activate bctproject
python -c "
import json, pymongo, sys

# --- UPDATE THIS with your connection string ---
URI = 'mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority'

client = pymongo.MongoClient(URI)
db = client['crs']
col = db['college data']

with open('../sql 123.json') as f:
    data = json.load(f)

col.drop()  # clear existing data
result = col.insert_many(data)
print(f'Inserted {len(result.inserted_ids)} colleges into crs.college data')
client.close()
"
```

### 3.6 Verify the import

After importing, you should have:
- **Database:** `crs`
- **Collection:** `college data`
- **Documents:** 15 (each with fields: `CollegeId`, `Name`, `Location`, `Type`, `ContactNumber`, `Email`, `HostelAvailability`, `Departments`, etc.)

---

## Step 4 — Configure Environment Variables

Edit the file `.env.mongodb` in the project root with your MongoDB connection details:

```bash
# .env.mongodb
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=crs
MONGODB_COLLECTION=college data
```

Replace `<username>` and `<password>` with your actual MongoDB Atlas credentials from Step 3.2.

> **Important:** The backend reads this file automatically on startup. Do **not** commit this file with real credentials to a public repo.

---

## Step 5 — Train the Models

The trained model files are **not included** in the repository. You must train both models before running the server.

### 5.1 Train the BERT Intent Classifier

This trains a BERT model to classify user messages into 12 intent categories (greeting, college_details, compare_colleges, etc.).

**Training data:** `data/intent_entity.json`
**Output:** `models/bert_intent_model/`

```bash
# From the project root:
cd backend

conda activate bctproject

python -m app.nlu.intent.train_bert_intent
```

**Optional arguments:**
```bash
python -m app.nlu.intent.train_bert_intent \
  --data_path ../data/intent_entity.json \
  --output_dir ../models/bert_intent_model \
  --epochs 8 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --tune   # run hyperparameter grid search (takes longer)
```

Training takes roughly 5–15 minutes depending on your hardware. When done, the `models/bert_intent_model/` directory will contain the model files.

### 5.2 Train the RoBERTa+CRF Entity Model

This trains a RoBERTa model with a CRF layer to extract entities (college names, courses, locations, budgets, etc.) using BIO tagging.

**Training data:** `data/bio_dataset.json`
**Output:** `models/crf_entity_model/`

```bash
# From the project root (still in backend/):
python -m app.nlu.entity.train_with_crf
```

**Optional arguments:**
```bash
python -m app.nlu.entity.train_with_crf \
  --data_path ../data/bio_dataset.json \
  --output_dir ../models/crf_entity_model \
  --epochs 15 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --patience 3 \
  --tune   # run hyperparameter grid search
```

Training takes roughly 10–25 minutes. When done, `models/crf_entity_model/` will contain `model.pt` and tokenizer files.

### 5.3 Verify models exist

After training both models, you should have:

```
models/
├── bert_intent_model/
│   ├── model.safetensors       # Model weights
│   ├── config.json
│   ├── label_mapping.json      # Intent label mapping
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
│
└── crf_entity_model/
    ├── model.pt                # CRF model weights
    ├── label_mappings.json     # BIO label mapping
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.json
```

---

## Step 6 — Run the Backend

```bash
cd backend
conda activate bctproject
python -m app.main
```

The server starts at **http://localhost:8000**. You should see:
```
🚀 Starting dialogue system...
✅ Dialogue system initialized successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify it is working:
```bash
curl http://localhost:8000/api/v1/health/detailed
```

---

## Step 7 — Run the Frontend

Open a **new terminal**:

```bash
cd frontend
npm install
npm start
```

The React app opens at **http://localhost:3000** and proxies API requests to the backend at port 8000.

---

## API Endpoints

### Chat

```bash
POST /api/v1/chat
Content-Type: application/json

{
    "message": "Tell me about Kathmandu University",
    "session_id": "optional-session-id"
}
```

### Health Check

```bash
GET /api/v1/health           # Basic
GET /api/v1/health/detailed  # Full system status (DB, models, etc.)
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Architecture Overview

```
User Message
    │
    ▼
┌─────────────────────┐
│   NLU Pipeline       │
│  ┌───────────────┐  │
│  │ BERT Intent    │──│──▶ Intent (12 labels)
│  │ Classifier     │  │
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │ RoBERTa+CRF   │──│──▶ Entities (9 types, BIO tags)
│  │ NER Extractor  │  │
│  └───────────────┘  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Dialogue Manager    │
│  • Intent locking    │
│  • Slot filling      │
│  • Fuzzy matching    │
│  • Context tracking  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Intent Handlers     │──▶ MongoDB Query ──▶ Format Response
│  (10 handlers)       │
└─────────────────────┘
    │
    ▼
  Response
```

**12 Intent Labels:** greeting, goodbye, college_details, compare_colleges, search_college, best_items_search, personalized_recommendation, recommend_with_constraints, hostel_query, contact_query, college_attribute_query, admission_process

**9 Entity Types:** COURSE, LOCATION, COLLEGE_TYPE, RANK, BUDGET, COLLEGE_NAME, COLLEGE_NAME_1, COLLEGE_NAME_2, ATTRIBUTE

**Database:** MongoDB Atlas — `crs` database, `college data` collection, 15 Nepal engineering colleges

---

## Troubleshooting

| Problem | Solution |
| ------- | -------- |
| `ModuleNotFoundError: No module named 'torchcrf'` | Run `pip install pytorch-crf` |
| `ModuleNotFoundError: No module named 'rapidfuzz'` | Run `pip install rapidfuzz` |
| MongoDB connection timeout | Check `.env.mongodb` URI, network access whitelist in Atlas |
| Models not found on startup | Train both models first (Step 5) |
| Frontend can't reach backend | Make sure backend is running on port 8000 |
| CUDA out of memory during training | Add `--batch_size 8` to reduce memory |
| `seqeval` not found during entity training | Run `pip install seqeval` |