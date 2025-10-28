import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

export default function Chatbot() {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "👋 Hello! I'm your TEM Career Assistant. How can I help you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const handleSend = async (e) => {
    e.preventDefault();
    const query = input.trim();
    if (!query) return;

    // Add user message
    setMessages((prev) => [...prev, { sender: "user", text: query }]);
    setInput("");
    setIsTyping(true);

    try {
      const res = await axios.post("http://127.0.0.1:8000/ask", { query });
      const answer =
        res.data.answer || "Sorry, I couldn't find that information.";
      setIsTyping(false);
      setMessages((prev) => [...prev, { sender: "bot", text: answer }]);
    } catch (err) {
      console.error(err);
      setIsTyping(false);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Error: Could not connect to the server." },
      ]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
    // Shift + Enter or Alt + Enter will create new line naturally
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4">
      <div className="w-full max-w-4xl bg-white rounded-2xl shadow-2xl overflow-hidden border border-slate-200 flex flex-col h-[85vh]">
        {/* Premium Header */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 px-6 py-4 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center backdrop-blur-sm">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                </div>
                <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-400 border-2 border-white rounded-full"></div>
              </div>
              <div>
                <h1 className="text-xl font-bold">TEM Career Assistant</h1>
                <p className="text-blue-100 text-sm">AI-powered career guidance</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <span className="text-sm text-blue-100">Online</span>
            </div>
          </div>
        </div>

        {/* Enhanced Chat Area */}
        <div className="flex-1 overflow-y-auto p-6 bg-gradient-to-b from-slate-50 to-white space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${
                msg.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[80%] rounded-2xl p-4 shadow-sm border ${
                  msg.sender === "user"
                    ? "bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-br-none border-blue-400"
                    : "bg-white text-slate-700 rounded-bl-none border-slate-200 shadow-md"
                }`}
              >
                <div className="flex items-start space-x-3">
                  {msg.sender === "bot" && (
                    <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                    </div>
                  )}
                  <div className="flex-1">
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                    <div className={`text-xs mt-2 ${msg.sender === "user" ? "text-blue-200" : "text-slate-400"}`}>
                      {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                  {msg.sender === "user" && (
                    <div className="w-8 h-8 bg-gradient-to-r from-slate-600 to-slate-700 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}

          {/* Enhanced Typing Indicator */}
          {isTyping && (
            <div className="flex justify-start">
              <div className="max-w-[80%] bg-white rounded-2xl rounded-bl-none p-4 shadow-md border border-slate-200">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse delay-75"></div>
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse delay-150"></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Premium Input Area */}
        <div className="border-t border-slate-200 bg-white p-6">
          <form onSubmit={handleSend} className="flex items-end space-x-4">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="Ask about colleges, courses, career paths... (Press Enter to send, Shift+Enter for new line)"
                rows="1"
                className="w-full p-4 pr-12 border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-slate-50 transition-all duration-200"
              />
              <div className="absolute right-3 bottom-3 flex space-x-2">
                <button
                  type="button"
                  className="p-2 text-slate-400 hover:text-slate-600 transition-colors"
                  title="Attach file"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                  </svg>
                </button>
              </div>
            </div>
            <button
              type="submit"
              disabled={!input.trim()}
              className="p-4 bg-gradient-to-r from-blue-600 to-indigo-700 text-white rounded-xl hover:from-blue-700 hover:to-indigo-800 transition-all duration-200 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              title="Send message"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
              </svg>
            </button>
          </form>
          <div className="mt-3 text-xs text-slate-500 text-center flex items-center justify-center space-x-4">
            <span className="flex items-center">
              <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-300 rounded">Enter</kbd>
              <span className="ml-1">to send</span>
            </span>
            <span className="flex items-center">
              <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-300 rounded">Shift</kbd>
              <span className="mx-1">+</span>
              <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-300 rounded">Enter</kbd>
              <span className="ml-1">for new line</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}