# College Recommendation System 🎓

A full-stack web application that helps students find the perfect engineering college using AI-powered recommendations and FAISS vector search.

## 🏗️ Architecture

```
┌────────────┐        WebSocket        ┌──────────────┐
│   React    │  <-------------------> │   FastAPI     │
│  Frontend  │                        │   Backend     │
└────────────┘                        └──────┬───────┘
                                              │
                                              │ Python call
                                              ▼
                                     ┌──────────────────┐
                                     │   AI Model       │
                                     │ (Embeddings/RAG) │
                                     └──────────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │ Vector DB / JSON │
                                     │ (FAISS + Data)   │
                                     └──────────────────┘
```

## ✨ Features

- **🤖 AI-Powered Chatbot**: Natural language processing with FAISS vector search
- **📊 Smart Recommendations**: Personalized college suggestions based on preferences  
- **💬 Real-time Chat**: WebSocket-based instant messaging
- **🔍 Comprehensive Search**: Query colleges by location, fees, courses, ratings, and more
- **📱 Responsive Design**: Works on desktop and mobile devices
- **⚡ Fast Performance**: Sub-second response times with vector similarity search

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- npm or yarn

### One-Command Setup
```bash
chmod +x setup.sh && ./setup.sh
```

### Manual Setup

#### Backend
```bash
cd backend
pip3 install -r requirements.txt
pip3 install faiss-cpu sentence-transformers pandas scikit-learn
python3 main.py
```

#### Frontend  
```bash
cd frontend
npm install
npm start
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🎯 Usage Examples

### Chat Queries
- "Where is Sagarmatha Engineering College located?"
- "Engineering colleges in Kathmandu under 10 lakhs"  
- "Does Pulchowk have hostel facilities?"
- "Email of Kathmandu University"
- "Scholarship opportunities at IOE"

### API Usage
```python
import requests

response = requests.post("http://localhost:8000/chat", 
                        json={"message": "Best engineering colleges in Nepal"})
print(response.json())
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance web framework
- **WebSockets**: Real-time bidirectional communication  
- **FAISS**: Vector similarity search (782 documents)
- **SentenceTransformers**: Text embeddings (all-MiniLM-L6-v2)
- **Python**: Core logic and AI model integration

### Frontend
- **React 18**: Modern UI library
- **WebSocket API**: Real-time messaging
- **CSS3**: Custom responsive styling
- **JavaScript ES6+**: Modern frontend development

### AI/Data
- **FAISS Vector Database**: 782 indexed documents
- **Sentence Transformers**: Semantic search capabilities
- **College Dataset**: 36+ colleges with comprehensive course information
- **RAG Architecture**: Retrieval-Augmented Generation for accurate responses

## 📊 Data Coverage

- **36+ Engineering Colleges** across Nepal
- **86+ Course Programs** with detailed information
- **Comprehensive Data**: Locations, fees, ratings, scholarships, internships
- **Contact Information**: Phone numbers, email addresses
- **Academic Details**: Pass rates, faculty ratios, admission processes

## 🚀 Performance

- **Sub-second Response Times**: Optimized FAISS indexing
- **782 Document Search**: Comprehensive knowledge base
- **Real-time Updates**: WebSocket-based instant messaging
- **Scalable Architecture**: Ready for production deployment

## 📁 Project Structure

```
college-recommendation-system/
├── backend/
│   ├── main.py              # FastAPI server with WebSocket
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── index.js        # React entry point
│   │   └── index.css       # Styling
│   ├── public/
│   │   └── index.html      # HTML template
│   └── package.json        # Node dependencies
├── stand_alone.py          # AI chatbot core logic
├── full_data.json          # College dataset
└── setup.sh               # Automated setup script
```

## 🔧 Development

### Run Backend in Development
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run Frontend in Development  
```bash
cd frontend
npm start
```

## 🌐 Deployment

### Backend (Production)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend (Build)
```bash
cd frontend
npm run build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎓 About

Developed as a final year project for BCT (Bachelor of Computer Technology) program. This system demonstrates the integration of modern web technologies with AI/ML for practical educational applications.

---

**Built with ❤️ for students seeking quality engineering education in Nepal**