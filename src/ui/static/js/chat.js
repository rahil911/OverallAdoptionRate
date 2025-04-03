/**
 * Overall Adoption Rate Chatbot - Chat Functionality
 * 
 * This script manages the chat interface functionality including:
 * - Sending and receiving messages
 * - Loading and error states
 * - Handling the chat history
 * - Managing UI interactions
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatArea = document.getElementById('chatArea');
    const resetButton = document.getElementById('resetButton');
    const toggleChartButton = document.getElementById('toggleChartButton');
    const chartArea = document.getElementById('chartArea');
    const typingIndicator = document.getElementById('typingIndicator');
    const exampleQueryButtons = document.querySelectorAll('.example-query-btn');
    
    // Templates
    const userMessageTemplate = document.getElementById('userMessageTemplate');
    const assistantMessageTemplate = document.getElementById('assistantMessageTemplate');
    const loadingTemplate = document.getElementById('loadingTemplate');
    const errorMessageTemplate = document.getElementById('errorMessageTemplate');
    
    // Chat state
    let isProcessing = false;
    let loadingElement = null;
    let chartVisible = false;
    
    /**
     * Initialize the chat interface and load conversation history if available
     */
    function initChat() {
        // Check if there's an existing conversation
        fetch('/api/conversation_history')
            .then(response => response.json())
            .then(data => {
                if (data.history && data.history.length > 0) {
                    // Render existing conversation
                    data.history.forEach(message => {
                        if (message.role === 'user') {
                            addUserMessage(message.content, message.timestamp);
                        } else if (message.role === 'assistant') {
                            addAssistantMessage(message.content, message.timestamp);
                        }
                    });
                    
                    // Scroll to the bottom
                    scrollToBottom();
                    
                    // Remove welcome message if we have history
                    const welcomeMessage = document.querySelector('.welcome-message');
                    if (welcomeMessage) {
                        welcomeMessage.remove();
                    }
                }
            })
            .catch(error => {
                console.error('Error loading conversation history:', error);
            });
    }
    
    /**
     * Send a user message to the chatbot API
     * @param {string} message - The user's message
     */
    function sendMessage(message) {
        if (isProcessing || !message.trim()) {
            return;
        }
        
        isProcessing = true;
        
        // Remove welcome message if it exists
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        // Add user message to UI
        addUserMessage(message);
        
        // Clear input
        userInput.value = '';
        
        // Show loading indicator
        showLoading();
        
        console.log('[DEBUG] Sending message to API:', message);
        
        // Send to API
        fetch('/api/message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        })
        .then(response => {
            console.log(`[DEBUG] API response status: ${response.status} ${response.statusText}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('[DEBUG] API returned chat response:', data);
            
            // Hide loading indicator
            hideLoading();
            
            // Add assistant response to UI
            addAssistantMessage(data.response);
            
            // Update chat state
            isProcessing = false;
        })
        .catch(error => {
            console.error('[DEBUG] Error processing message:', error);
            
            // Hide loading indicator
            hideLoading();
            
            // Show error message
            addErrorMessage(`There was an error processing your request: ${error.message}`);
            
            // Update chat state
            isProcessing = false;
        });
    }
    
    /**
     * Reset the conversation
     */
    function resetConversation() {
        fetch('/api/reset_conversation', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Clear chat area
                chatArea.innerHTML = '';
                
                // Add welcome message back
                addWelcomeMessage();
                
                // Reset chart if needed
                if (typeof resetChart === 'function') {
                    resetChart();
                }
            }
        })
        .catch(error => {
            console.error('Error resetting conversation:', error);
        });
    }
    
    /**
     * Toggle the chart visibility
     */
    function toggleChart() {
        chartVisible = !chartVisible;
        
        if (chartVisible) {
            chartArea.style.display = 'block';
            toggleChartButton.innerHTML = '<i class="fas fa-comments"></i> Toggle Chat';
            
            // Update chart if needed
            if (typeof updateChart === 'function') {
                updateChart();
            }
        } else {
            chartArea.style.display = 'none';
            toggleChartButton.innerHTML = '<i class="fas fa-chart-line"></i> Toggle Chart';
        }
    }
    
    /**
     * Add a user message to the chat area
     * @param {string} message - The user's message
     * @param {string} timestamp - Optional timestamp for the message
     */
    function addUserMessage(message, timestamp = null) {
        const clone = userMessageTemplate.content.cloneNode(true);
        const messageContent = clone.querySelector('.message-content p');
        const messageTimestamp = clone.querySelector('.message-timestamp');
        
        messageContent.textContent = message;
        
        if (timestamp) {
            messageTimestamp.textContent = timestamp;
        } else {
            messageTimestamp.textContent = getCurrentTimestamp();
        }
        
        chatArea.appendChild(clone);
        scrollToBottom();
    }
    
    /**
     * Add an assistant message to the chat area
     * @param {string} message - The assistant's message
     * @param {string} timestamp - Optional timestamp for the message
     */
    function addAssistantMessage(message, timestamp = null) {
        const clone = assistantMessageTemplate.content.cloneNode(true);
        const messageContent = clone.querySelector('.message-content p');
        const messageTimestamp = clone.querySelector('.message-timestamp');
        
        // Support for markdown/links/code formatting
        messageContent.innerHTML = formatMessage(message);
        
        if (timestamp) {
            messageTimestamp.textContent = timestamp;
        } else {
            messageTimestamp.textContent = getCurrentTimestamp();
        }
        
        chatArea.appendChild(clone);
        scrollToBottom();
    }
    
    /**
     * Add an error message to the chat area
     * @param {string} message - The error message
     */
    function addErrorMessage(message) {
        const clone = errorMessageTemplate.content.cloneNode(true);
        const messageContent = clone.querySelector('.message-content p');
        
        messageContent.textContent = message;
        
        chatArea.appendChild(clone);
        scrollToBottom();
    }
    
    /**
     * Add welcome message to the chat area
     */
    function addWelcomeMessage() {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <h2>Welcome to the Overall Adoption Rate Chatbot</h2>
            <p>Ask questions about adoption rates, trends, and insights. For example:</p>
            <ul class="example-queries">
                <li><button class="example-query-btn">What is our current adoption rate?</button></li>
                <li><button class="example-query-btn">How has our adoption rate changed since last quarter?</button></li>
                <li><button class="example-query-btn">What caused the spike in adoption in March 2023?</button></li>
                <li><button class="example-query-btn">What will our adoption rate be next month?</button></li>
                <li><button class="example-query-btn">How can we improve our adoption rate?</button></li>
            </ul>
        `;
        
        chatArea.appendChild(welcomeDiv);
        
        // Reattach event listeners to new buttons
        const newQueryButtons = welcomeDiv.querySelectorAll('.example-query-btn');
        newQueryButtons.forEach(button => {
            button.addEventListener('click', function() {
                sendMessage(this.textContent);
            });
        });
    }
    
    /**
     * Show loading indicator
     */
    function showLoading() {
        // Add loading message to UI
        loadingElement = loadingTemplate.content.cloneNode(true);
        chatArea.appendChild(loadingElement);
        scrollToBottom();
        
        // Show typing indicator
        typingIndicator.style.display = 'block';
    }
    
    /**
     * Hide loading indicator
     */
    function hideLoading() {
        // Remove loading messages
        const loadingMessages = document.querySelectorAll('.message.assistant-message.loading');
        loadingMessages.forEach(el => el.remove());
        
        // Hide typing indicator
        typingIndicator.style.display = 'none';
    }
    
    /**
     * Format the message content to support links and code blocks
     * @param {string} message - The message to format
     * @returns {string} - The formatted message HTML
     */
    function formatMessage(message) {
        if (!message) return '';
        
        // Replace URLs with clickable links
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        let formattedMessage = message.replace(urlRegex, '<a href="$1" target="_blank">$1</a>');
        
        // Format code blocks
        formattedMessage = formattedMessage.replace(/```([^`]+)```/g, '<pre>$1</pre>');
        
        return formattedMessage;
    }
    
    /**
     * Get the current timestamp in a readable format
     * @returns {string} - Formatted timestamp
     */
    function getCurrentTimestamp() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    /**
     * Scroll to the bottom of the chat area
     */
    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }
    
    /**
     * Handle chat form submission
     */
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = userInput.value.trim();
        
        if (message && !isProcessing) {
            sendMessage(message);
        }
    });
    
    /**
     * Handle example query button clicks
     */
    exampleQueryButtons.forEach(button => {
        button.addEventListener('click', function() {
            sendMessage(this.textContent);
        });
    });
    
    /**
     * Handle reset button click
     */
    resetButton.addEventListener('click', resetConversation);
    
    /**
     * Handle chart toggle button click
     */
    toggleChartButton.addEventListener('click', toggleChart);
    
    // Initialize the chat
    initChat();
}); 