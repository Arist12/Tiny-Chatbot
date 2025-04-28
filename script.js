document.addEventListener('DOMContentLoaded', function() {
    userMessages = [];
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearButton = document.querySelector('.action-button');

    // Sample bot responses to randomly choose from

    // Format time for timestamp
    function formatTime() {
        const now = new Date();
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    // Function to create typing indicator
    function createTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'bot-message', 'typing-indicator');

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');

        const icon = document.createElement('i');
        icon.classList.add('fas', 'fa-robot');
        avatar.appendChild(icon);

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');

        const dots = document.createElement('div');
        dots.classList.add('typing-dots');

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dots.appendChild(dot);
        }

        messageContent.appendChild(dots);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        return messageDiv;
    }

    // Function to add a message to the chat
    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');

        const icon = document.createElement('i');
        icon.classList.add('fas');
        icon.classList.add(isUser ? 'fa-user' : 'fa-robot');
        avatar.appendChild(icon);

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');

        const paragraph = document.createElement('p');
        paragraph.textContent = message;
        messageContent.appendChild(paragraph);

        // Add timestamp
        const messageFooter = document.createElement('div');
        messageFooter.classList.add('message-footer');

        const timestamp = document.createElement('span');
        timestamp.classList.add('timestamp');
        timestamp.textContent = formatTime();
        messageFooter.appendChild(timestamp);

        messageContent.appendChild(messageFooter);

        if (isUser) {
            messageDiv.appendChild(messageContent);
            messageDiv.appendChild(avatar);
        } else {
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
        }

        chatMessages.appendChild(messageDiv);

        // Scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to handle user input
    function handleUserInput() {
        const message = userInput.value.trim();
        if (message) {
            // Add user message
            userMessages.push(message);
            addMessage(message, true);

            // Clear input field
            userInput.value = '';

            // Add typing indicator
            const typingIndicator = createTypingIndicator();
            chatMessages.appendChild(typingIndicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Simulate bot "typing" with a delay
            setTimeout(() => {
                // // Remove typing indicator
                // chatMessages.removeChild(typingIndicator);

                // // Get random bot response
                // const randomIndex = Math.floor(Math.random() * botResponses.length);
                // const botResponse = botResponses[randomIndex];

                // // Add bot response
                // addMessage(botResponse);
                const fullText = userMessages.join(' ');
                getMBTIPrediction(fullText).then(prediction => {
                    // Remove typing indicator
                    chatMessages.removeChild(typingIndicator);

                    // Add bot response
                    addMessage(prediction);
                });
            }, 1500);
        }
    }

    // Function to clear chat
    function clearChat() {
        // Keep only the first message (welcome message)
        while (chatMessages.children.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        
        // Clear the saved user messages
        userMessages = []; 
        // Update the timestamp of the first message
        const firstMessageFooter = chatMessages.querySelector('.message-footer');
        if (firstMessageFooter) {
            const timestamp = firstMessageFooter.querySelector('.timestamp');
            if (timestamp) {
                timestamp.textContent = formatTime();
            }
        }
    }

    async function getMBTIPrediction(text) {
        const response = await fetch('http://localhost:8000/predict-mbti', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        return data.chatbot_response;
    }

    // Send button click event
    sendButton.addEventListener('click', handleUserInput);

    // Clear chat button click event
    clearButton.addEventListener('click', clearChat);

    // Enter key press event
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default to avoid newline
            handleUserInput();
        }
    });

    // Auto-resize textarea as user types
    userInput.addEventListener('input', function() {
        // Reset height to auto to get the correct scrollHeight
        this.style.height = 'auto';

        // Calculate new height (min is original height, max is 120px)
        const newHeight = Math.min(this.scrollHeight, 120);

        // Set the new height
        this.style.height = newHeight + 'px';
    });

    // Focus on input when page loads
    userInput.focus();
});
