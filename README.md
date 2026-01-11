# College Recommendation Dialogue System

A production-ready AI dialogue system for college information and recommendations, built with FastAPI and following Rasa/DeepPavlov architecture patterns.

## 🏗️ Architecture

```
backend/app/                  # Main application
├── api/                      # FastAPI routes (chat, health)
├── nlu/                      # Natural Language Understanding
│   ├── intent/              # BERT intent classification
│   └── entity/              # RoBERTa entity extraction
├── context/                  # Dialogue state tracking
├── policy/                   # Rule-based dialogue policy
├── actions/                  # Action handlers (college search, etc.)
├── services/                 # Business logic layer
├── repositories/             # Data access layer (MongoDB)
├── response/                 # Response formatting & templates
├── schemas/                  # Pydantic data models
├── utils/                    # Configuration, logging, constants
├── dialogue_manager.py       # Main orchestrator
└── main.py                   # FastAPI entry point

data/                         # Training data
├── entity/                   # Entity training data
├── intent/                   # Intent training data
└── full_data.json           # College dataset

models/                       # Trained ML models
├── bert_intent_model/        # BERT intent classifier
├── roberta_entity_model/     # RoBERTa entity extractor
└── faiss_index/             # Vector similarity search

frontend/                     # React frontend (optional)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Conda environment: `bctproject`
- MongoDB Atlas connection

### Running the Backend

```bash
cd backend
conda activate bctproject
python -m app.main
```

The API will be available at:
- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

### API Endpoints

#### Chat
```bash
POST /api/v1/chat
{
    "message": "Tell me about Kathmandu University",
    "session_id": "optional_session_id"
}
```

#### Health Check
```bash
GET /api/v1/health/detailed
```

## 🧠 Dialogue Flow

The system follows a clean **Rasa-inspired architecture**:

1. **NLU Processing** → Intent + Entity extraction
2. **Context Tracking** → Session state management  
3. **Policy Planning** → Action selection
4. **Action Execution** → College search, information retrieval
5. **Response Generation** → Natural language responses

## 📊 Models

- **Intent Classification**: BERT with 23 intent classes
- **Entity Extraction**: RoBERTa with 25 entity types
- **College Database**: 36 colleges in MongoDB Atlas

## 🔧 Configuration

Configuration is managed through `backend/app/utils/config.py`:

- **Database**: MongoDB connection settings
- **Models**: Model paths and confidence thresholds  
- **API**: CORS, ports, workers
- **Dialogue**: Session timeout, max turns

## 📝 Development

The codebase follows clean architecture principles:
- **Separation of Concerns**: Each layer has specific responsibilities
- **Dependency Injection**: Services receive dependencies
- **Type Safety**: Full Pydantic schema validation
- **Async Support**: Non-blocking operations
- **Error Handling**: Comprehensive error management
- **Logging**: Structured logging throughout