import React, { useState } from 'react';
import './App.css';

function ChatHistory({ chats, onSelectChat, onNewChat, selectedChatId }) {
  return (
    <div className="chat-history">
      <div className="chat-history-header">
        <h2>Chats</h2>
        <button onClick={onNewChat} className="new-chat-btn">+ New Chat</button>
      </div>
      <ul className="chat-list">
        {chats.map((chat) => (
          <li
            key={chat.id}
            className={chat.id === selectedChatId ? 'selected' : ''}
            onClick={() => onSelectChat(chat.id)}
          >
            {chat.title}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Avatar({ sender }) {
  const isUser = sender === 'user';
  return (
    <div className={`avatar ${isUser ? 'avatar-user' : 'avatar-assistant'}`}> 
      {isUser ? '🧑' : '🤖'}
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.sender === 'user';
  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'} animated-bubble`}>
      <Avatar sender={message.sender} />
      <div className="bubble-and-time">
        <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>{message.text}</div>
        {isUser && <div className="timestamp">{message.timestamp}</div>}
      </div>
    </div>
  );
}

function TypingIndicator({ show }) {
  if (!show) return null;
  return (
    <div className="typing-indicator">
      <span className="dot" />
      <span className="dot" />
      <span className="dot" />
      <span className="typing-text">Assistant is typing...</span>
    </div>
  );
}

function ChatWindow({ chat, messages, onSendMessage, typing }) {
  const [input, setInput] = useState('');

  const handleSend = (e) => {
    e.preventDefault();
    if (input.trim()) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  return (
    <div className="chat-window">
      <div className="chat-header">
        <h3>{chat ? chat.title : 'Select or create a chat'}</h3>
      </div>
      <div className="chat-body">
        {chat && messages.length === 0 && (
          <div className="empty-chat">Start the conversation!</div>
        )}
        {chat && messages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} />
        ))}
        <TypingIndicator show={typing} />
      </div>
      {chat && (
        <form className="chat-input-row" onSubmit={handleSend}>
          <input
            className="chat-input"
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={e => setInput(e.target.value)}
            autoFocus
          />
          <button className="send-btn" type="submit">Send</button>
        </form>
      )}
    </div>
  );
}

function NewChatModal({ isOpen, onClose, onCreate }) {
  const [chatName, setChatName] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (chatName.trim()) {
      onCreate(chatName.trim());
      setChatName('');
    }
  };

  if (!isOpen) return null;
  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>New Chat</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Enter chat name"
            value={chatName}
            onChange={e => setChatName(e.target.value)}
            autoFocus
          />
          <div className="modal-actions">
            <button type="button" onClick={onClose} className="modal-cancel">Cancel</button>
            <button type="submit" className="modal-create">Create</button>
          </div>
        </form>
      </div>
    </div>
  );
}

function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function App() {
  const [chats, setChats] = useState([
    { id: 1, title: 'Chat 1' },
    { id: 2, title: 'Chat 2' },
  ]);
  const [selectedChatId, setSelectedChatId] = useState(chats[0]?.id || null);
  const [modalOpen, setModalOpen] = useState(false);
  const [typing, setTyping] = useState(false);
  const [messagesByChat, setMessagesByChat] = useState({
    1: [
      { sender: 'assistant', text: 'Hello! How can I help you today?', timestamp: getCurrentTime() },
      { sender: 'user', text: 'Hi! Can you tell me about your features?', timestamp: getCurrentTime() },
      { sender: 'assistant', text: 'Of course! I can answer questions, help with tasks, and more.', timestamp: getCurrentTime() },
    ],
    2: [],
  });

  const handleNewChat = () => {
    setModalOpen(true);
  };

  const handleCreateChat = (chatName) => {
    const newId = chats.length ? Math.max(...chats.map(c => c.id)) + 1 : 1;
    const newChat = { id: newId, title: chatName };
    setChats([...chats, newChat]);
    setSelectedChatId(newId);
    setMessagesByChat({ ...messagesByChat, [newId]: [] });
    setModalOpen(false);
  };

  const handleSelectChat = (id) => {
    setSelectedChatId(id);
  };

  const handleSendMessage = (text) => {
    if (!selectedChatId) return;
    setMessagesByChat(prev => ({
      ...prev,
      [selectedChatId]: [...(prev[selectedChatId] || []), { sender: 'user', text, timestamp: getCurrentTime() }],
    }));
    setTyping(true);
    setTimeout(() => {
      setMessagesByChat(prev => ({
        ...prev,
        [selectedChatId]: [...(prev[selectedChatId] || []), { sender: 'assistant', text: 'This is a sample assistant reply.', timestamp: getCurrentTime() }],
      }));
      setTyping(false);
    }, 1200);
  };

  const selectedChat = chats.find(c => c.id === selectedChatId);
  const messages = selectedChat ? messagesByChat[selectedChatId] || [] : [];

  return (
    <div className="app-container">
      <ChatHistory
        chats={chats}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        selectedChatId={selectedChatId}
      />
      <ChatWindow
        chat={selectedChat}
        messages={messages}
        onSendMessage={handleSendMessage}
        typing={typing}
      />
      <NewChatModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onCreate={handleCreateChat}
      />
    </div>
  );
}

export default App;
