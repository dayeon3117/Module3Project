import React, { useState } from "react";
import axios from "axios";

const Chatbot = () => {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");

  const BACKEND_URL = "https://nba-rag-chatbot.onrender.com";

  const handleSubmit = async (e) => {
      e.preventDefault();
      setResponse("Thinking...");
  
      try {
          const res = await axios.post(`${BACKEND_URL}/ask`, {
              query: question,
          });
          setResponse(res.data.response);
      } catch (error) {
          setResponse("Error fetching response. Make sure the backend is running.");
      }
  };

  return (
    <div className="chat-container">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Ask the NBA chatbot..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="submit">Ask</button>
      </form>
      <div className="response">
        <strong>Response:</strong>
        <p>{response}</p>
      </div>
    </div>
  );
};

export default Chatbot;
