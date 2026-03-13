# College Recommendation System (CRS)

An AI-powered chatbot for Nepal engineering college information and recommendations. Built with **FastAPI** (backend), **React** (frontend), **MongoDB Atlas** (database), and custom-trained **BERT** + **RoBERTa+CRF** NLU models.

> **Live API:** `https://api.rabiaryal.com.np` (exposed via Cloudflare Tunnel)
> **API Key:** `demo-secret-2026` (pass in every request as `x-api-key` header)

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

---

---

# API Integration Guide

Everything you need to call this API from any frontend or tool.

---

## Base URLs

| Environment | URL |
|---|---|
| Local development | `http://localhost:8000` |
| Live (Cloudflare Tunnel) | `https://api.rabiaryal.com.np` |

---

## Authentication

Every request to `/api/v1/chat` must include this header:

```
x-api-key: demo-secret-2026
```

Missing or wrong key returns:
```json
HTTP 401 Unauthorized
{ "detail": "Missing or incorrect x-api-key" }
```

> To change the key, edit `DEMO_API_KEY` in `backend/app/api/auth.py` and restart the server.

---

## Endpoints

### POST `/api/v1/chat` — Send a message

**Request headers:**

| Header | Value | Required |
|---|---|---|
| `x-api-key` | `demo-secret-2026` | ✅ Yes |
| `Content-Type` | `application/json` | ✅ Yes |

**Request body:**

```json
{
  "session_id": "abc-123",
  "message": "show me colleges under 5 lakh"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✅ Yes | The user's message (1–1000 chars) |
| `session_id` | string | ⚠️ Recommended | Tracks conversation context across turns. Omitting it generates a new session each time (no memory). |

**Response body:**

```json
{
  "message": "Here are colleges within your budget of Rs. 5,00,000...",
  "session_id": "abc-123",
  "intent": "recommend_with_constraints",
  "entities": {
    "budget": 500000
  },
  "confidence": 0.91,
  "timestamp": "2026-03-11T10:30:00",
  "debug_info": {}
}
```

| Field | Type | Description |
|---|---|---|
| `message` | string | The chatbot's reply (may contain Markdown) |
| `session_id` | string | Echo of the session ID (save this for follow-up messages) |
| `intent` | string | Detected intent label |
| `entities` | object | Extracted slots (budget, rank, course, college_name, etc.) |
| `confidence` | float | Intent confidence score (0.0–1.0) |
| `timestamp` | string | ISO 8601 datetime |
| `debug_info` | object | Internal pipeline info (can be ignored) |

---

### GET `/api/v1/health` — Basic health check

No auth required.

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2026-03-11T10:30:00"
}
```

### GET `/api/v1/health/detailed` — Full system status

No auth required. Returns DB connection status, model load status, uptime.

---

## Session Management

The `session_id` is how the server remembers what a user said previously.

```
Turn 1:  "show me colleges under 5 lakh"   → bot asks: "which course?"
Turn 2:  "computer"                         → bot returns recommendations

Both turns MUST use the same session_id.
```

**Rules:**
- Generate the `session_id` **once** when the user opens the chat (e.g. `crypto.randomUUID()` in JS).
- Send that same ID with **every** message in the conversation.
- To reset the conversation, either send `"message": "clear"` or generate a new `session_id`.

---

## Usage Examples

### cURL

```bash
# First message
curl -X POST https://api.rabiaryal.com.np/api/v1/chat \
     -H "x-api-key: demo-secret-2026" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my-session-1", "message": "show me colleges under 5 lakh"}'

# Follow-up (same session_id — bot remembers budget)
curl -X POST https://api.rabiaryal.com.np/api/v1/chat \
     -H "x-api-key: demo-secret-2026" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my-session-1", "message": "computer"}'

# Reset conversation
curl -X POST https://api.rabiaryal.com.np/api/v1/chat \
     -H "x-api-key: demo-secret-2026" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my-session-1", "message": "clear"}'
```

---

### Postman

1. **Method:** `POST`
2. **URL:** `https://api.rabiaryal.com.np/api/v1/chat`
3. **Headers tab:**
   - `x-api-key` → `demo-secret-2026`
   - `Content-Type` → `application/json`
4. **Body tab** → raw → JSON:
   ```json
   {
     "session_id": "test-session-1",
     "message": "compare KEC and IOE"
   }
   ```

**Auto-capture session_id between requests** — in the **Tests** tab of your first request:
```javascript
var res = pm.response.json();
pm.environment.set("session_id", res.session_id);
```

Then in every subsequent body use `{{session_id}}`:
```json
{
  "session_id": "{{session_id}}",
  "message": "which one has hostel?"
}
```

---

### React / JavaScript (fetch)

```javascript
// chatApi.js — copy this into your React project

const API_BASE = "https://api.rabiaryal.com.np";
const API_KEY  = "demo-secret-2026";

// Call this once when the chat window opens
export function createSessionId() {
  return crypto.randomUUID();
}

export async function sendMessage(sessionId, message) {
  const response = await fetch(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
    }),
  });

  if (response.status === 401) {
    throw new Error("Invalid API key");
  }
  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }

  return response.json(); // returns the full ChatResponse object
}
```

**Usage in a React component:**

```jsx
import { useState, useRef } from "react";
import { createSessionId, sendMessage } from "./chatApi";

export default function ChatWidget() {
  const sessionId = useRef(createSessionId()); // fixed for this session
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);

  async function handleSend() {
    if (!input.trim()) return;
    const userMsg = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", text: userMsg }]);
    setLoading(true);

    try {
      const data = await sendMessage(sessionId.current, userMsg);
      setMessages(prev => [...prev, { role: "bot", text: data.message }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: "bot", text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div>
        {messages.map((m, i) => (
          <p key={i}><strong>{m.role}:</strong> {m.text}</p>
        ))}
        {loading && <p>...</p>}
      </div>
      <input value={input} onChange={e => setInput(e.target.value)}
             onKeyDown={e => e.key === "Enter" && handleSend()} />
      <button onClick={handleSend}>Send</button>
    </div>
  );
}
```

---

### Python (requests)

```python
import requests
import uuid

API_BASE = "https://api.rabiaryal.com.np"
API_KEY  = "demo-secret-2026"
HEADERS  = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json",
}

session_id = str(uuid.uuid4())  # generate once per conversation

def chat(message: str) -> str:
    response = requests.post(
        f"{API_BASE}/api/v1/chat",
        headers=HEADERS,
        json={"session_id": session_id, "message": message},
    )
    response.raise_for_status()
    return response.json()["message"]

# Example conversation
print(chat("show me colleges under 5 lakh"))
print(chat("computer"))          # follow-up — bot remembers budget
print(chat("which have hostel")) # follow-up — bot remembers budget + course
```

---

### Next.js / Node.js (server-side API route)

```javascript
// pages/api/chat.js  (or app/api/chat/route.js for App Router)

const API_BASE = "https://api.rabiaryal.com.np";
const API_KEY  = "demo-secret-2026"; // keep this server-side only

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  const { session_id, message } = req.body;

  const upstream = await fetch(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
    },
    body: JSON.stringify({ session_id, message }),
  });

  const data = await upstream.json();
  res.status(upstream.status).json(data);
}
```

> This proxies the request through your Next.js server so the API key is never exposed to the browser.

---

### Flutter / Dart

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:uuid/uuid.dart';

const String _apiBase = 'https://api.rabiaryal.com.np';
const String _apiKey  = 'demo-secret-2026';

class CrsApiService {
  final String sessionId = const Uuid().v4(); // one per conversation

  Future<String> sendMessage(String message) async {
    final uri = Uri.parse('$_apiBase/api/v1/chat');
    final response = await http.post(
      uri,
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': _apiKey,
      },
      body: jsonEncode({
        'session_id': sessionId,
        'message': message,
      }),
    );

    if (response.statusCode == 401) throw Exception('Invalid API key');
    if (response.statusCode != 200) throw Exception('Server error');

    final data = jsonDecode(response.body);
    return data['message'] as String;
  }
}
```

---

## What the Bot Can Answer

| Example message | Intent triggered |
|---|---|
| `hello` / `hi` | `greeting` |
| `tell me about KEC` | `college_details` |
| `compare IOE and KU` | `compare_colleges` |
| `show colleges in Kathmandu` | `search_college` |
| `top rated colleges` | `best_items_search` |
| `recommend colleges for rank 500, budget 6 lakh` | `personalized_recommendation` |
| `colleges under 5 lakh for computer` | `recommend_with_constraints` |
| `does KEC have hostel?` | `hostel_query` |
| `contact number of Sagarmatha college` | `contact_query` |
| `what is the fee of KU?` | `college_attribute_query` |
| `how to get admission in KEC?` | `admission_process` |
| `clear` | resets session context |

---

## Intent & Entity Reference

**12 Intent labels:**
`greeting`, `goodbye`, `college_details`, `compare_colleges`, `search_college`,
`best_items_search`, `personalized_recommendation`, `recommend_with_constraints`,
`hostel_query`, `contact_query`, `college_attribute_query`, `admission_process`

**9 Entity types extracted from messages:**

| Entity | Example |
|---|---|
| `COLLEGE_NAME` | "KEC", "Kathmandu University" |
| `COURSE` | "Computer", "Civil", "BE Computer" |
| `LOCATION` | "Kathmandu", "Lalitpur" |
| `BUDGET` | "5 lakh", "600000" |
| `RANK` | "500", "rank 1200" |
| `COLLEGE_TYPE` | "government", "private" |
| `ATTRIBUTE` | "fee", "rating", "hostel" |
| `COLLEGE_NAME_1` | first college in comparisons |
| `COLLEGE_NAME_2` | second college in comparisons |

**Budget normalization:**

| User input | Stored as |
|---|---|
| `"5 lakh"` | `500000` |
| `"600000"` | `600000` |
| `"7"` *(bare number < 1000)* | `700000` |
| `"1.5 lakh"` | `150000` |

---

## Running the Backend

```bash
# 1. Activate environment
conda activate bctproject

# 2. Start server
cd backend
python -m app.main
```

Server starts at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

**With Cloudflare Tunnel** (run in a separate terminal):
```bash
cloudflared tunnel run
```

Both must be running simultaneously for the live URL to work.

---

## Changing the API Key

Edit the one line in `backend/app/api/auth.py`:

```python
DEMO_API_KEY = os.getenv("DEMO_API_KEY", "demo-secret-2026")
```

Or set an environment variable before starting the server:
```bash
export DEMO_API_KEY="my-new-key"
python -m app.main
```

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

**Database:** MongoDB Atlas — `crs` database, `college data` collection, 15 Nepal engineering colleges

---

## Troubleshooting

| Problem | Solution |
| ------- | -------- |
| `401 Unauthorized` | Check `x-api-key` header value — must be `demo-secret-2026` |
| Follow-up messages lose context | Ensure same `session_id` is sent in every message |
| `ModuleNotFoundError: No module named 'torchcrf'` | Run `pip install pytorch-crf` |
| `ModuleNotFoundError: No module named 'rapidfuzz'` | Run `pip install rapidfuzz` |
| MongoDB connection timeout | Check `.env.mongodb` URI, whitelist IP in Atlas Network Access |
| Models not found on startup | Train both models first (Step 5) |
| Frontend can't reach backend | Ensure backend is running on port 8000 |
| CUDA out of memory during training | Add `--batch_size 8` to training command |
| `seqeval` not found during entity training | Run `pip install seqeval` |