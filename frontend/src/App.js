import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './App.css';

function MarkdownMessage({ text }) {
  return (
    <ReactMarkdown
      children={text}
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          return !inline ? (
            <SyntaxHighlighter
              style={oneDark}
              language={match ? match[1] : 'python'}
              PreTag="div"
              customStyle={{ borderRadius: '12px', margin: '18px 0', fontSize: '1.08rem', background: '#23232a', padding: '18px' }}
              {...props}
            >
              {String(children).replace(/\n$/, '')}
            </SyntaxHighlighter>
          ) : (
            <code className={className} {...props} style={{ background: '#23232a', padding: '2px 6px', borderRadius: '4px' }}>
              {children}
            </code>
          );
        },
      }}
    />
  );
}

function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <span className="dot" />
      <span className="dot" />
      <span className="dot" />
      <span className="typing-text">Assistant is typing...</span>
    </div>
  );
}

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('chat_history');
    return saved ? JSON.parse(saved) : [];
  });
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    localStorage.setItem('chat_history', JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMessage = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input }),
      });
      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: data.answer || 'No response.' },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: 'Error connecting to backend.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="gpt-bg">
      <div className="gpt-sidebar">
        <div className="gpt-sidebar-header">
          <span className="gpt-logo">🧠</span>
          <span className="gpt-title">ChatGPT</span>
        </div>
        <button className="gpt-sidebar-btn">+ New chat</button>
        <div className="gpt-sidebar-section">
          <div className="gpt-sidebar-item">Search chats</div>
          <div className="gpt-sidebar-item">Library</div>
          <div className="gpt-sidebar-item">Sora</div>
        </div>
        <div className="gpt-sidebar-section gpt-sidebar-bottom">
          <div className="gpt-sidebar-user">Avirup Dasgupta<br /><span className="gpt-sidebar-user-type">Free</span></div>
        </div>
      </div>
      <div className="gpt-main">
        <div className="gpt-main-content">
          {messages.length === 0 && !loading ? (
            <div className="gpt-welcome">
              <h1>What can I help with?</h1>
              <div className="gpt-big-input-row wide">
                <span className="gpt-attach-icon">📎</span>
                <input
                  className="gpt-big-input wide"
                  placeholder="Ask anything"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSend(e)}
                  disabled={loading}
                />
                <span className="gpt-big-input-icon">🎤</span>
                <button className="gpt-big-input-send" onClick={handleSend} disabled={loading}> <span role="img" aria-label="send">📤</span> </button>
              </div>
            </div>
          ) : (
            <div className="gpt-chat-area gpt-chat-area-flat">
              {messages.map((msg, idx) => (
                <div key={idx} className={`gpt-chat-message-flat ${msg.sender} ${msg.sender === 'user' ? 'right' : ''}`}> {/* user messages right-aligned */}
                  {msg.sender === 'user' ? (
                    <span>{msg.text}</span>
                  ) : (
                    <MarkdownMessage text={msg.text} />
                  )}
                </div>
              ))}
              {loading && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
        {messages.length > 0 || loading ? (
          <form className="gpt-main-input-row wide" onSubmit={handleSend}>
            <span className="gpt-attach-icon">📎</span>
            <input
              className="gpt-main-input wide"
              placeholder="Send a message..."
              value={input}
              onChange={e => setInput(e.target.value)}
              disabled={loading}
            />
            <span className="gpt-main-input-icon">🎤</span>
            <button className="gpt-main-input-send" type="submit" disabled={loading}> <span role="img" aria-label="send">📤</span> </button>
          </form>
        ) : null}
      </div>
    </div>
  );
}

export default App;
