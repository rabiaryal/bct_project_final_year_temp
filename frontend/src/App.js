import React, { useState, useEffect, useRef } from 'react';
import {
    Search,
    Filter,
    Home,
    BookOpen,
    Building2,
    DollarSign,
    Send,
    X,
    Menu,
    MapPin,
    GraduationCap,
    Wifi,
    WifiOff,
    Loader,
    Bot,
    User,
    MessageCircle
} from 'lucide-react';
import './index.css';

const App = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [isTyping, setIsTyping] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [sessionId, setSessionId] = useState(null);

    // Filter states
    const [selectedLocation, setSelectedLocation] = useState('');
    const [selectedCourse, setSelectedCourse] = useState('');
    const [selectedCollegeType, setSelectedCollegeType] = useState('');
    const [maxFee, setMaxFee] = useState(1500000);
    const [needsHostel, setNeedsHostel] = useState(false);
    const [needsScholarship, setNeedsScholarship] = useState(false);

    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    // API Base URL
    const API_BASE_URL = 'http://localhost:8000';

    // Filter options
    const locations = ['Kathmandu', 'Lalitpur', 'Bhaktapur', 'Pokhara', 'Chitwan', 'Dharan', 'Butwal', 'Dhulikhel'];
    const courses = ['Civil Engineering', 'Computer Engineering', 'Electronics Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Architecture'];

    useEffect(() => {
        checkConnection();
        generateSessionId();

        // Set up periodic connection check
        const interval = setInterval(checkConnection, 30000); // Check every 30 seconds
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const generateSessionId = () => {
        const newSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        setSessionId(newSessionId);
    };

    const checkConnection = async () => {
        setIsConnecting(true);
        try {
            const response = await fetch(`${API_BASE_URL}/api/v1/health`);
            if (response.ok) {
                setIsConnected(true);
                console.log('✅ Backend connected successfully');
            } else {
                setIsConnected(false);
                console.log('❌ Backend health check failed');
            }
        } catch (error) {
            setIsConnected(false);
            console.error('❌ Backend connection error:', error);
        }
        setIsConnecting(false);
    };

    const sendMessage = async (messageText = null) => {
        const textToSend = messageText || inputMessage.trim();

        if (!textToSend || !isConnected) {
            return;
        }

        const userMessage = {
            id: Date.now(),
            type: 'user',
            content: textToSend,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputMessage('');
        setIsTyping(true);

        try {
            const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: textToSend,
                    session_id: sessionId
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Received response:', data);

            const botMessage = {
                id: Date.now() + 1,
                type: 'bot',
                content: data.message,
                timestamp: new Date(),
                intent: data.intent,
                entities: data.entities,
                confidence: data.confidence,
                metadata: {
                    intent: data.intent,
                    confidence: data.confidence,
                    entities: data.entities
                }
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Error sending message:', error);
            const errorMessage = {
                id: Date.now() + 1,
                type: 'bot',
                content: 'Sorry, I encountered an error processing your message. Please try again.',
                timestamp: new Date(),
                metadata: { error: true }
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsTyping(false);
        }
    };

    const handleRecommendClick = () => {
        if (!isConnected) return;

        let queryParts = ['recommend colleges'];
        if (selectedCourse) queryParts.push(`for ${selectedCourse}`);
        if (selectedLocation) queryParts.push(`in ${selectedLocation}`);
        if (selectedCollegeType) queryParts.push(`${selectedCollegeType.toLowerCase()}`);
        if (maxFee < 1500000) queryParts.push(`under ${maxFee / 100000} lakh`);
        if (needsHostel) queryParts.push('with hostel');
        if (needsScholarship) queryParts.push('with scholarship');

        const query = queryParts.join(' ');
        sendMessage(`🔍 ${query}`);
    };

    const clearFilters = () => {
        setSelectedLocation('');
        setSelectedCourse('');
        setSelectedCollegeType('');
        setMaxFee(1500000);
        setNeedsHostel(false);
        setNeedsScholarship(false);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        sendMessage();
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const clearChat = () => {
        setMessages([]);
        generateSessionId(); // Generate new session for fresh start
    };

    const formatTimestamp = (timestamp) => {
        return new Date(timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getConnectionStatus = () => {
        if (isConnected) {
            return {
                text: "Backend Ready",
                icon: <Wifi size={16} />,
                className: "status-connected"
            };
        } else if (isConnecting) {
            return {
                text: "Checking...",
                icon: <Loader size={16} className="animate-spin" />,
                className: "status-connecting"
            };
        } else {
            return {
                text: "Backend Offline",
                icon: <WifiOff size={16} />,
                className: "status-disconnected"
            };
        }
    };

    const formatMessage = (message) => {
        return message
            .replace(/•/g, '•')
            .replace(/\n/g, '\n');
    };

    const connectionStatus = getConnectionStatus();

    return (
        <div className="app">
            {/* Side Panel for Filters */}
            <div className={`filters-panel ${isSidebarOpen ? 'open' : 'collapsed'}`}>
                <div className="filters-header">
                    <Filter size={20} />
                    {isSidebarOpen && <span>Filters</span>}
                    {isSidebarOpen && (
                        <button
                            className="clear-filters-btn"
                            onClick={clearFilters}
                            title="Clear all filters"
                        >
                            <X size={16} />
                        </button>
                    )}
                </div>

                {isSidebarOpen && (
                    <div className="filters-content">
                        <div className="filters-row">
                            <div className="filter-group">
                                <label><MapPin size={16} /> Location</label>
                                <select
                                    value={selectedLocation}
                                    onChange={(e) => setSelectedLocation(e.target.value)}
                                >
                                    <option value="">Any Location</option>
                                    {locations.map(loc => (
                                        <option key={loc} value={loc}>{loc}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="filter-group">
                                <label><BookOpen size={16} /> Course</label>
                                <select
                                    value={selectedCourse}
                                    onChange={(e) => setSelectedCourse(e.target.value)}
                                >
                                    <option value="">Any Course</option>
                                    {courses.map(course => (
                                        <option key={course} value={course}>{course}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="filter-group">
                                <label><Building2 size={16} /> Type</label>
                                <select
                                    value={selectedCollegeType}
                                    onChange={(e) => setSelectedCollegeType(e.target.value)}
                                >
                                    <option value="">Any Type</option>
                                    <option value="Public">Public</option>
                                    <option value="Private">Private</option>
                                </select>
                            </div>
                        </div>

                        <div className="filters-row">
                            <div className="filter-group fee-filter">
                                <label><DollarSign size={16} /> Max Fee: ₹{(maxFee / 100000).toFixed(1)} Lakh</label>
                                <input
                                    type="range"
                                    min="0"
                                    max="1500000"
                                    step="50000"
                                    value={maxFee}
                                    onChange={(e) => setMaxFee(Number(e.target.value))}
                                    className="fee-slider"
                                />
                                <div className="fee-labels">
                                    <span>₹0</span>
                                    <span>₹15L</span>
                                </div>
                            </div>
                        </div>

                        <div className="filters-row">
                            <div className="filter-group checkbox-group">
                                <label className="checkbox-label">
                                    <input
                                        type="checkbox"
                                        checked={needsHostel}
                                        onChange={(e) => setNeedsHostel(e.target.checked)}
                                    />
                                    <span className="checkmark"></span>
                                    <Home size={16} /> Hostel Required
                                </label>
                                <label className="checkbox-label">
                                    <input
                                        type="checkbox"
                                        checked={needsScholarship}
                                        onChange={(e) => setNeedsScholarship(e.target.checked)}
                                    />
                                    <span className="checkmark"></span>
                                    <GraduationCap size={16} /> Scholarship Available
                                </label>
                            </div>

                            <button
                                className="recommend-btn"
                                onClick={handleRecommendClick}
                                disabled={!isConnected}
                            >
                                <Search size={16} />
                                Get Recommendations
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Main Chat Area */}
            <div className="main-chat-area">
                {/* Header */}
                <div className="chat-header-main">
                    <button
                        className="sidebar-toggle"
                        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                    >
                        {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
                    </button>
                    <GraduationCap size={28} className="header-icon" />
                    <div className="header-text">
                        <h1>College Recommendation System</h1>
                        <p>Find your perfect college in Nepal</p>
                    </div>
                    <div className={`connection-status ${connectionStatus.className}`}>
                        {connectionStatus.icon}
                        <span>{connectionStatus.text}</span>
                    </div>
                </div>

                {/* Messages */}
                <div className="messages-container">
                    {messages.map((message) => (
                        <div
                            key={message.id}
                            className={`message message-${message.type}`}
                        >
                            <div className="message-header">
                                {message.type === 'user' ? (
                                    <User size={20} className="message-icon user-icon" />
                                ) : (
                                    <Bot size={20} className="message-icon bot-icon" />
                                )}
                                <span className="message-time">
                                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                            </div>
                            <div className="message-content">
                                {formatMessage(message.content)}
                            </div>
                        </div>
                    ))}

                    {/* Typing indicator */}
                    {isTyping && (
                        <div className="message typing-indicator">
                            <div className="message-header">
                                <Bot size={20} className="message-icon bot-icon" />
                                <span className="message-time">now</span>
                            </div>
                            <div className="message-content">
                                <div className="typing-dots">
                                    <span></span>
                                    <span></span>
                                    <span></span>
                                </div>
                                Thinking...
                            </div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="input-container">
                    <form onSubmit={handleSubmit} className="input-form">
                        <div className="input-wrapper">
                            <MessageCircle size={20} className="input-icon" />
                            <textarea
                                ref={inputRef}
                                value={inputMessage}
                                onChange={(e) => setInputMessage(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder={isConnected ? "Ask about colleges, courses, fees, or anything else..." : "Connecting to chatbot..."}
                                className="input-field"
                                disabled={!isConnected}
                                rows={1}
                            />
                            <button
                                type="submit"
                                disabled={!isConnected || !inputMessage.trim()}
                                className="send-button"
                                title="Send message"
                            >
                                <Send size={16} />
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default App;