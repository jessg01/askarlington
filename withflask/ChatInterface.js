import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './ChatInterface.css';

// Get the API URL from environment variables or use default
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [citizenId, setCitizenId] = useState(`user_${Math.random().toString(36).substring(2, 10)}`); // Generate random user ID
  const messagesEndRef = useRef(null);
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Reset chat if there's an error
  const resetChat = () => {
    setMessages([]);
    setInitialized(false);
    setCitizenId(`user_${Math.random().toString(36).substring(2, 10)}`);
    setError(null);
    setInput('');
    setIsLoading(false);
  };

  // Initialize chat with a welcome message from the server
  useEffect(() => {
    const initializeChat = async () => {
      if (!initialized) {
        setIsLoading(true);
        setError(null);
        try {
          console.log(`Initializing chat with citizen_id: ${citizenId}`);
          const response = await axios.post(`${API_URL}/api/chat`, {
            message: 'hello',
            citizen_id: citizenId
          }, {
            timeout: 20000,
            withCredentials: true // Important for session cookies
          });
          
          console.log('Received initialization response:', response.data);
          
          if (response.data.error) {
            setMessages([{ text: `Error: ${response.data.error}`, sender: 'system' }]);
            setError(response.data.error);
          } else if (response.data.response) {
            setMessages([{ text: response.data.response, sender: 'bot' }]);
            setInitialized(true);
          }
        } catch (error) {
          console.error('Error initializing chat:', error);
          const errorMessage = error.response?.data?.error || 
                               error.message || 
                               'Failed to connect to the server';
          
          setMessages([{ 
            text: `Connection error: ${errorMessage}. Try refreshing the page.`, 
            sender: 'system' 
          }]);
          setError(errorMessage);
        } finally {
          setIsLoading(false);
        }
      }
    };
    
    initializeChat();
  }, [citizenId, initialized]);

  // Function to truncate long bot responses
  const formatBotResponse = (text) => {
    // Set a maximum character limit for bot responses
    const MAX_LENGTH = 1500;
    
    if (!text) return "No response received";
    
    if (text && text.length > MAX_LENGTH) {
      return text.substring(0, MAX_LENGTH) + "... [Response truncated due to length]";
    }
    
    return text;
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    if (error) {
      setMessages(prev => [...prev, { 
        text: "Please reset the chat before continuing.", 
        sender: 'system' 
      }]);
      return;
    }

    const userMessage = input;
    setInput('');
    
    // Add user message to chat
    setMessages(prev => [...prev, { text: userMessage, sender: 'user' }]);
    setIsLoading(true);

    try {
      console.log(`Sending message to ${citizenId}: ${userMessage}`);
      const payload = {
        message: userMessage,
        citizen_id: citizenId
      };

      const response = await axios.post(`${API_URL}/api/chat`, payload, {
        timeout: 60000, // 60-second timeout
        withCredentials: true // Important for session cookies
      });
      
      console.log('Received response:', response.data);
      
      if (response.data.error) {
        setMessages(prev => [...prev, { text: `Error: ${response.data.error}`, sender: 'system' }]);
        setError(response.data.error);
      } else {
        // Format and potentially truncate the bot response
        const formattedResponse = formatBotResponse(response.data.response);
        
        setMessages(prev => [...prev, { text: formattedResponse, sender: 'bot' }]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      let errorMessage = 'Sorry, I encountered an error. Please try again later.';
      
      // Handle specific errors
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'The request timed out. The AI might be taking too long to respond.';
      } else if (error.response) {
        errorMessage = `Server error (${error.response.status}): ${
          error.response.data?.error || 'Unknown error'
        }`;
      }
      
      setMessages(prev => [...prev, { 
        text: errorMessage, 
        sender: 'system' 
      }]);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h3>City Chatbot</h3>
        {error && (
          <button className="reset-button" onClick={resetChat}>
            Reset Chat
          </button>
        )}
      </div>
      
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h2>Welcome to the City Chatbot!</h2>
            <p>Ask me anything about city services, events, or information.</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <div className="message-content">{msg.text}</div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="message bot">
            <div className="message-content loading">
              <span>.</span><span>.</span><span>.</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input-form" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={error ? "Reset chat to continue..." : "Type your message here..."}
          disabled={isLoading || error}
        />
        <button type="submit" disabled={isLoading || !input.trim() || error}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface; 