# Overall Adoption Rate Chatbot - UI Components

This directory contains the UI components for the Overall Adoption Rate Chatbot.

## Structure

- `app.py`: Main Flask application
- `api_endpoints.py`: API endpoints for fetching data
- `templates/`: HTML templates
- `static/`: Static assets (CSS, JS, images)

## Components

### Chat Interface
- Chat input and message display
- Loading states and typing indicators
- Error messages and fallback UI states
- Example queries for easy access

### Chart Visualization
- Dynamic chart visualization using Plotly.js
- Time period and metric filters
- Highlights for peaks and valleys
- Responsive design for various screen sizes

## Running the UI

To run the UI locally:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```
   python -m src.ui.app
   ```

3. Access the UI in your browser at:
   ```
   http://localhost:5000
   ```

## API Endpoints

- `/api/message`: Process a chat message
- `/api/conversation_history`: Get the conversation history
- `/api/reset_conversation`: Reset the conversation
- `/api/chart_data`: Get data for the chart visualization

## Mobile Responsiveness

The UI is designed to be responsive across different device types:
- Desktop: Full layout with side-by-side chat and chart
- Tablet: Adjustable layout with toggleable chart
- Mobile: Stacked layout optimized for smaller screens 