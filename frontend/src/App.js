import React, { useState, useEffect, useRef } from 'react';
import './index.css';

const App = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [isTyping, setIsTyping] = useState(false);

    // Filter states
    const [selectedLocation, setSelectedLocation] = useState('');
    const [selectedCourse, setSelectedCourse] = useState('');
    const [selectedCollegeType, setSelectedCollegeType] = useState('');
    const [maxFee, setMaxFee] = useState(1500000);
    const [needsHostel, setNeedsHostel] = useState(false);
    const [needsScholarship, setNeedsScholarship] = useState(false);

    const messagesEndRef = useRef(null);
    const wsRef = useRef(null);
    const inputRef = useRef(null);

    // Filter options
    const locations = ['Kathmandu', 'Lalitpur', 'Bhaktapur', 'Pokhara', 'Chitwan', 'Dharan', 'Butwal', 'Dhulikhel'];
    const courses = ['Civil Engineering', 'Computer Engineering', 'Electronics Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Architecture'];
    const collegeTypes = ['Public', 'Private'];

    useEffect(() => {
        connectWebSocket();
        // eslint-disable-next-line react-hooks/exhaustive-deps
        return () => {
            if (wsRef.current) wsRef.current.close();
        };
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const connectWebSocket = () => {
        setIsConnecting(true);

        // Determine WebSocket URL - force localhost for development
        const wsUrl = 'ws://localhost:8000/ws';

        console.log('Attempting to connect to:', wsUrl);

        try {
            wsRef.current = new WebSocket(wsUrl);

            wsRef.current.onopen = () => {
                setIsConnected(true);
                setIsConnecting(false);
                console.log('✅ WebSocket connected successfully');
            };

            wsRef.current.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('Received message:', data);

                    if (data.type === 'typing') {
                        setIsTyping(true);
                        return;
                    }

                    setIsTyping(false);

                    const newMessage = {
                        id: Date.now() + Math.random(),
                        type: data.type,
                        content: data.message,
                        timestamp: new Date(),
                        userQuery: data.user_query,
                        metadata: data.metadata || {}
                    };

                    setMessages(prev => [...prev, newMessage]);
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            };

            wsRef.current.onclose = (event) => {
                setIsConnected(false);
                setIsConnecting(false);
                setIsTyping(false);
                console.log('❌ WebSocket disconnected:', event.code, event.reason);

                // Auto-reconnect after 3 seconds
                setTimeout(() => {
                    if (!isConnected) {
                        console.log('🔄 Attempting to reconnect...');
                        connectWebSocket();
                    }
                }, 3000);
            };

            wsRef.current.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                setIsConnected(false);
                setIsConnecting(false);
            };

        } catch (error) {
            console.error('Error creating WebSocket:', error);
            setIsConnecting(false);
        }
    };

    const sendMessage = (messageText = null) => {
        const textToSend = messageText || inputMessage.trim();

        if (!textToSend || !isConnected || !wsRef.current) {
            return;
        }

        const userMessage = {
            id: Date.now(),
            type: 'user',
            content: textToSend,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        wsRef.current.send(JSON.stringify({ message: textToSend }));
        setInputMessage('');
    };

    const handleRecommendClick = () => {
        if (!isConnected || !wsRef.current) return;

        let queryParts = ['recommend colleges'];
        if (selectedCourse) queryParts.push(`for ${selectedCourse}`);
        if (selectedLocation) queryParts.push(`in ${selectedLocation}`);
        if (selectedCollegeType) queryParts.push(`${selectedCollegeType.toLowerCase()}`);
        if (maxFee < 1500000) queryParts.push(`under ${maxFee / 100000} lakh`);
        if (needsHostel) queryParts.push('with hostel');
        if (needsScholarship) queryParts.push('with scholarship');

        const query = queryParts.join(' ');

        const userMessage = {
            id: Date.now(),
            type: 'user',
            content: `🔍 ${query}`,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        wsRef.current.send(JSON.stringify({
            message: query,
            filters: {
                location: selectedLocation || null,
                course: selectedCourse || null,
                college_type: selectedCollegeType || null,
                max_fee: maxFee,
                hostel_required: needsHostel,
                scholarship_needed: needsScholarship
            }
        }));
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

    const getConnectionStatus = () => {
        if (isConnected) {
            return { text: "🟢 Connected", className: "status-connected" };
        } else if (isConnecting) {
            return { text: "🟡 Connecting...", className: "status-connecting" };
        } else {
            return { text: "🔴 Disconnected", className: "status-disconnected" };
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
            <div className="chat-container">
                {/* Header */}
                <div className="chat-header">
                    <h1>🎓 College Recommendation Chatbot</h1>
                    <p>Powered by XGBoost & AI</p>
                </div>

                {/* Connection Status */}
                <div className={`connection-status ${connectionStatus.className}`}>
                    {connectionStatus.text}
                </div>

                {/* Filter Panel */}
                <div className="filters-panel">
                    <div className="filters-row">
                        <div className="filter-group">
                            <label>📍 Location</label>
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
                            <label>📚 Course</label>
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
                            <label>🏛️ Type</label>
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
                            <label>💰 Max Fee: ₹{(maxFee / 100000).toFixed(1)} Lakh</label>
                            <input
                                type="range"
                                min="0"
                                max="1500000"
                                step="50000"
                                value={maxFee}
                                onChange={(e) => setMaxFee(Number(e.target.value))}
                                className="fee-slider"
                            />
                        </div>

                        <div className="filter-group checkbox-group">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={needsHostel}
                                    onChange={(e) => setNeedsHostel(e.target.checked)}
                                />
                                🏠 Hostel
                            </label>
                            <label>
                                <input
                                    type="checkbox"
                                    checked={needsScholarship}
                                    onChange={(e) => setNeedsScholarship(e.target.checked)}
                                />
                                🎓 Scholarship
                            </label>
                        </div>

                        <button
                            className="recommend-btn"
                            onClick={handleRecommendClick}
                            disabled={!isConnected}
                        >
                            🔍 Get Recommendations
                        </button>
                    </div>
                </div>

                {/* Messages */}
                <div className="messages-container">
                    {messages.map((message) => (
                        <div
                            key={message.id}
                            className={`message message-${message.type}`}
                        >
                            <div className="message-content">
                                {formatMessage(message.content)}
                            </div>
                        </div>
                    ))}

                    {/* Typing indicator */}
                    {isTyping && (
                        <div className="message typing-indicator">
                            <div className="message-content">
                                Bot is typing...
                            </div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="input-container">
                    <form onSubmit={handleSubmit} className="input-form">
                        <textarea
                            ref={inputRef}
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder={isConnected ? "Ask about colleges, courses, fees..." : "Connecting to chatbot..."}
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
                            ➤
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default App;