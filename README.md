# College Recommendation System (CRS)

An AI-powered conversational system for Nepal engineering college discovery, comparison, and personalized recommendations.

This project combines:
- FastAPI backend with session-aware dialogue management
- React frontend chat interface
- MongoDB Atlas data store
- Custom NLU stack: BERT intent classification + RoBERTa + CRF entity extraction

## Table of Contents

1. Overview
2. Key Features
3. System Architecture
4. Tech Stack
5. Repository Structure
6. Prerequisites
7. Quick Start (Clone -> Install -> Run)
8. Detailed Setup
9. Model Training
10. Build and Deployment Notes
11. Backend-Only Deployment via Cloudflared
12. Connect Any Frontend via API
13. API Reference
14. End-to-End Working Process
15. Intents and Entities
16. Troubleshooting

## Overview

CRS is a chatbot-style recommendation platform focused on engineering admissions in Nepal. Users can ask natural language questions such as:
- Find colleges by course, location, type, rank, and budget
- Compare two colleges side by side
- Get contact, hostel, and admission information
- Receive personalized recommendations with safety labels based on rank gap

The backend preserves multi-turn conversation context via a session ID so follow-up queries work naturally.

## Key Features

- Conversational NLU pipeline (intent + entity extraction)
- Multi-turn slot filling and context carryover
- 12 supported intents and 9 core entity types
- Hybrid recommendation engine:
  - MongoDB hard filtering for candidate retrieval
  - Python scoring/reranking for recommendation quality
- Personalized safety classification: SAFE, MODERATE, RISKY
- API-key protected chat endpoint
- Frontend chat UI with persistent browser session ID
- Detailed health and diagnostics endpoints

## System Architecture

```text
User/Frontend
    -> FastAPI /api/v1/chat
    -> Dialogue Manager
       -> Intent Classifier (BERT)
       -> Entity Extractor (RoBERTa + CRF)
       -> Slot Manager (normalize + validate + context)
       -> Intent Router
          -> Intent Handler
             -> Query Builder (MongoDB pipeline)
             -> MongoDB Atlas
             -> Scorer (for recommendation intents)
             -> Formatter
    -> Structured chat response
```

## Tech Stack

### Backend
- Python 3.11+
- FastAPI, Uvicorn
- Motor + PyMongo
- PyTorch + Transformers
- pytorch-crf
- RapidFuzz

### Frontend
- React 18
- react-scripts
- lucide-react icons

### Data and Models
- MongoDB Atlas
- BERT intent model (`models/bert_intent_model`)
- RoBERTa + CRF entity model (`models/crf_entity_model`)

## Repository Structure

```text
bct_final_year_project/
|- backend/
|  |- app/
|  |  |- api/                 # chat, health, auth
|  |  |- context/             # slot manager + dialogue context
|  |  |- core/                # query builder, scorer, slot_filler
|  |  |- handlers/            # intent-specific handlers + router
|  |  |- nlu/
|  |  |  |- intent/           # BERT runtime + training
|  |  |  |- entity/           # RoBERTa+CRF runtime + training
|  |  |- repositories/        # MongoDB data access
|  |  |- templates/           # intent templates
|  |  |- utils/               # config, logger, formatter
|  |  |- dialogue_manager.py  # end-to-end conversation pipeline
|  |  |- main.py              # FastAPI entrypoint
|  |- requirements.txt
|- frontend/
|  |- src/App.js              # chat UI + API integration
|  |- package.json
|- data/                       # datasets
|- models/                     # trained artifacts (local)
|- docs/                       # reports and graphs
|- README.md
```

## Prerequisites

- Git
- Python 3.11+
- Conda (recommended)
- Node.js 18+
- npm 9+
- MongoDB Atlas account (M0 free tier is enough)

Optional:
- CUDA/MPS-capable GPU for faster model training

## Quick Start (Clone -> Install -> Run)

```bash
git clone <your-repo-url>
cd bct_final_year_project

# Backend
conda create -n bctproject python=3.11 -y
conda activate bctproject
cd backend
pip install -r requirements.txt

# Frontend (new terminal)
cd ../frontend
npm install
```

Create `.env.mongodb` in project root:

```bash
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=crs
MONGODB_COLLECTION=college data
DEMO_API_KEY=demo-secret-2026
```

Run backend:

```bash
cd ../backend
conda activate bctproject
python -m app.main
```

Run frontend (new terminal):

```bash
cd frontend
npm start
```

Local URLs:
- Backend: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- Frontend: http://localhost:3000

## Detailed Setup

### 1. MongoDB Atlas Setup

1. Create Atlas cluster and DB user.
2. Add your IP in Network Access.
3. Create database: `crs`.
4. Create collection: `college data`.
5. Import dataset from `sql 123.json`.

### 1.1 How MongoDB Is Connected in This Project

MongoDB is connected from the backend using these components:

1. Environment values are loaded from `.env.mongodb`.
2. Config values are read in `backend/app/utils/config.py`.
3. Async Mongo client is created in `backend/app/repositories/mongo_client.py` using Motor.
4. Connection is initialized during app startup in `backend/app/main.py` via dialogue manager lifecycle.
5. Query handlers use the shared collection object to run aggregation pipelines.

Connection flow:

```text
.env.mongodb
   -> AppConfig (database config)
   -> MongoRepository.connect()
   -> AsyncIOMotorClient(MONGODB_URI)
   -> database = client[MONGODB_DB]
   -> collection = database[MONGODB_COLLECTION]
```

### 1.2 How a New User Should Set Up MongoDB

Create `.env.mongodb` at project root:

```bash
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=crs
MONGODB_COLLECTION=college data
DEMO_API_KEY=demo-secret-2026
```

Then import the JSON dataset:

```bash
mongoimport \
  --uri "mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/crs" \
  --collection "college data" \
  --file "sql 123.json" \
  --jsonArray
```

If you prefer GUI, import `sql 123.json` through MongoDB Compass into:
- Database: `crs`
- Collection: `college data`

### 1.3 Verify MongoDB Setup

1. Start backend:

```bash
cd backend
conda activate bctproject
python -m app.main
```

2. Check health endpoint:

```bash
curl http://localhost:8000/api/v1/health/detailed
```

3. Confirm MongoDB status is connected in the response.

4. Send a chat test request:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: demo-secret-2026" \
  -d '{"session_id":"mongo-test-1","message":"show me colleges in kathmandu"}'
```

If this returns college data, MongoDB setup is complete.

### 2. Install Optional Training Dependencies

If you plan to retrain models:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn seqeval
```

### 3. Verify Backend Health

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/health/detailed
```

## Model Training

Trained models are expected under `models/`.

### Train Intent Model (BERT)

```bash
cd backend
conda activate bctproject
python -m app.nlu.intent.train_bert_intent \
  --data_path ../data/intent_entity.json \
  --output_dir ../models/bert_intent_model
```

### Train Entity Model (RoBERTa + CRF)

```bash
cd backend
conda activate bctproject
python -m app.nlu.entity.train_with_crf \
  --data_path ../data/bio_dataset.json \
  --output_dir ../models/crf_entity_model
```

### Model Artifacts Used at Runtime

- `models/bert_intent_model/`
- `models/crf_entity_model/`

Note:
- An alternate `models/roberta_entity_model/` artifact may exist for experimentation/testing, but runtime config points to `crf_entity_model`.

## Build and Deployment Notes

### Frontend Production Build

```bash
cd frontend
npm run build
```

### Backend Production Serving

Current app can be run via:

```bash
python -m app.main
```

For production, use a process manager and secure environment variables.

### API Key

Chat endpoint requires `x-api-key` header.

Default key is loaded from:
- Environment variable `DEMO_API_KEY`
- Fallback: `demo-secret-2026`

## Backend-Only Deployment via Cloudflared

This system is designed so the backend can run independently, and any frontend can connect through HTTP APIs.

Deployment pattern:

```text
Any Frontend (React / Next.js / Flutter / Mobile App)
            |
            | HTTPS requests
            v
Public URL (Cloudflared Tunnel)
            |
            | forwards traffic
            v
FastAPI Backend (localhost:8000)
            |
            v
MongoDB + Models
```

How Cloudflared is used here:

1. Backend runs locally on port `8000`.
2. Cloudflared creates a secure outbound tunnel from your machine to Cloudflare.
3. Cloudflare provides a public HTTPS URL (for example, `https://api.rabiaryal.com.np`).
4. Public traffic to that URL is forwarded to local backend (`http://localhost:8000`).
5. No frontend hosting is required on the same server. Backend API becomes reusable by multiple clients.

Run backend + tunnel:

```bash
# Terminal 1
cd backend
conda activate bctproject
python -m app.main

# Terminal 2
cloudflared tunnel run
```

Notes:
- Cloudflared command depends on your tunnel setup and credentials.
- Keep API key validation enabled so public endpoint is protected.

## Connect Any Frontend via API

Because the backend is exposed as a standard HTTP API, any frontend can connect if it can send HTTPS requests.

Required integration rules:

1. Use backend base URL:
   - Local: `http://localhost:8000`
   - Public (Cloudflared): `https://api.rabiaryal.com.np`
2. Send `x-api-key` header on chat requests.
3. Send a persistent `session_id` per conversation for context-aware follow-ups.
4. Call `POST /api/v1/chat` with JSON body.

Minimal client request contract:

```json
{
  "session_id": "any-unique-id",
  "message": "show me civil colleges under 7 lakh"
}
```

This enables:
- React web app
- Next.js app
- Flutter mobile app
- Desktop client
- Postman/cURL testing

## API Reference

### Base URLs

- Local: `http://localhost:8000`
- Live: `https://api.rabiaryal.com.np`

### Authentication

Required header for chat:

```text
x-api-key: <your-api-key>
```

### Endpoints

1. `POST /api/v1/chat`
2. `GET /api/v1/health`
3. `GET /api/v1/health/detailed`
4. `GET /api/v1/chat/session/{session_id}`
5. `DELETE /api/v1/chat/session/{session_id}`

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: demo-secret-2026" \
  -d '{"session_id":"demo-session-1","message":"recommend civil colleges under 7 lakhs"}'
```

### Example Response

```json
{
  "message": "Here are the top recommendations matching your constraints...",
  "session_id": "demo-session-1",
  "intent": "recommend_with_constraints",
  "entities": {
    "COURSE": "civil",
    "BUDGET": "7 lakhs"
  },
  "confidence": 0.93,
  "timestamp": "2026-03-29T10:00:00",
  "debug_info": {}
}
```

## End-to-End Working Process

This is the full runtime flow for each user message:

1. Frontend sends message with `session_id` to `/api/v1/chat`.
2. API layer validates `x-api-key`.
3. Dialogue manager runs NLU:
   - BERT predicts intent
   - RoBERTa+CRF extracts entities
4. Post-NLU fixes are applied:
   - keyword override for low-confidence intent
   - entity correction (LOCATION -> COLLEGE_NAME when needed)
5. Slot manager updates context:
   - maps entities to slots
   - normalizes values (budget, rank, course aliases)
   - validates required slots
6. If required slots are missing, follow-up question is returned.
7. If actionable, router dispatches to relevant intent handler.
8. Handler builds MongoDB aggregation pipeline and fetches candidates.
9. For recommendation intents, candidates are reranked by Python scorer.
10. Formatter creates user-facing response text.
11. Response with intent, entities, confidence, debug info is returned.

## Intents and Entities

### Supported Intents

- greeting
- goodbye
- search_college
- best_items_search
- recommend_with_constraints
- personalized_recommendation
- compare_colleges
- college_details
- college_attribute_query
- hostel_query
- contact_query
- admission_process

### Core Entity Types

- COURSE
- LOCATION
- COLLEGE_TYPE
- RANK
- BUDGET
- HOSTEL
- COLLEGE_NAME
- COLLEGE_NAME_1
- COLLEGE_NAME_2
- ATTRIBUTE
- RATING

## Troubleshooting

| Problem | Fix |
| --- | --- |
| 401 Unauthorized | Check `x-api-key` header and `DEMO_API_KEY` value |
| Models not loading | Ensure model folders exist under `models/` and paths in config are correct |
| MongoDB timeout | Verify URI, credentials, DB access, and Atlas Network Access |
| Follow-up context lost | Keep same `session_id` for all turns in one conversation |
| `ModuleNotFoundError: torchcrf` | `pip install pytorch-crf` |
| `ModuleNotFoundError: rapidfuzz` | `pip install rapidfuzz` |
| Frontend cannot call backend | Ensure backend is running on port 8000 and frontend proxy is active |
| OOM during training | Reduce `--batch_size` and train on CPU/GPU with sufficient memory |

## Acknowledgements

This project was developed as a final-year engineering project focused on practical, explainable conversational recommendations for Nepal's engineering college selection process.