# AI Chatbot Interface

A modern, responsive chatbot web interface inspired by platforms like Claude and ChatGPT. This project provides a clean, user-friendly frontend that can be integrated with any language model backend.

## Usage
To start the chatbot, first make sure to download fastapi and uvicorn using following command:

```
pip install fastapi uvicorn
```

Then start the backend server using
```
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Lastly, start the frontend chat using liver server in vscode on ```index.html```
## Features

- **Clean UI Design**: Modern interface with responsive layout for all devices
- **Real-time Chat**: Fully functional chat interface with message history
- **User Interaction Features**:
  - Auto-expanding text input area
  - Send messages with Enter key or button click
  - Typing indicators showing when the bot is "responding"
  - Message timestamps
  - Clear chat history function
- **Visual Enhancements**:
  - Distinct user and bot message styling
  - Animated message transitions
  - Customizable avatars and colors

## Integrating Your Custom LLM Backend

The current implementation uses random predefined responses. To connect your own LLM:

1. **Replace the Random Response Logic**: In `script.js`, modify the `handleUserInput()` function:

```javascript
function handleUserInput() {
    const message = userInput.value.trim();
    if (message) {
        // Add user message
        addMessage(message, true);

        // Clear input field
        userInput.value = '';

        // Add typing indicator
        const typingIndicator = createTypingIndicator();
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Call your LLM API
        fetchBotResponse(message)
            .then(response => {
                // Remove typing indicator
                chatMessages.removeChild(typingIndicator);

                // Add bot response
                addMessage(response);
            })
            .catch(error => {
                // Handle errors
                console.error("Error:", error);
                chatMessages.removeChild(typingIndicator);
                addMessage("Sorry, I encountered an error. Please try again.");
            });
    }
}

// Function to call your LLM API
async function fetchBotResponse(userMessage) {
    // Replace with your LLM API endpoint
    const apiUrl = 'https://your-llm-api-endpoint.com/chat';

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer YOUR_API_KEY' // If required
            },
            body: JSON.stringify({
                message: userMessage,
                // Any other parameters your API needs
            })
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();
        return data.response; // Adjust according to your API response format
    } catch (error) {
        throw error;
    }
}
```

2. **Message History**: For context-aware responses, you may need to send conversation history to your LLM:

```javascript
// Track conversation history
const conversationHistory = [];

function addToHistory(message, isUser) {
    conversationHistory.push({
        role: isUser ? 'user' : 'assistant',
        content: message
    });
}

// Then update your message handling to include history
function handleUserInput() {
    // ...existing code...

    addToHistory(message, true);

    // Send full history to the LLM API
    fetchBotResponse(conversationHistory)
        .then(response => {
            // ...existing code...
            addToHistory(response, false);
        });
}
```

3. **Advanced Features**: Consider adding streaming responses, markdown rendering, code highlighting, or other features based on your LLM's capabilities
