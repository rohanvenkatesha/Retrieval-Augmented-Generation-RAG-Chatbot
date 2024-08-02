To install and run your Retrieval Augmented Generation (RAG) chatbot, follow these steps:

## Overview

The **RAG Chatbot** is a custom chatbot designed to analyze PDFs and images, allowing users to ask questions based on the content of these files. The chatbot uses Streamlit for the interface, LangChain for text processing, FAISS for vector storage, and Google Generative AI for generating responses.

## Features

- **PDF Text Analysis**: Extracts and processes text from uploaded PDF files, allowing users to ask detailed questions about the content.
- **Image Analysis**: Supports text extraction and analysis from images (JPG, JPEG, PNG) using Google Generative AI.
- **Conversational Memory**: Maintains conversational context for more coherent and relevant responses.
- **Customizable Prompts**: Users can tailor the chatbot’s behavior by modifying prompt templates.

## Installation

### Prerequisites

- Python 3.7 or higher
- [Google API Key](https://console.cloud.google.com/): Required for Google Generative AI.

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rohanvenkatesha/Retrieval-Augmented-Generation-RAG-Chatbot.git
   cd Retrieval-Augmented-Generation-RAG-Chatbot
   ```

2. **Create a Virtual Environment**:
   It's recommended to use a virtual environment to avoid conflicts:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add your Google API Key:
   ```text
   GOOGLE_API_KEY=<your_key>
   ```

4. **Install Dependencies**:
   Install all required Python packages using:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

After setting up, you can run the chatbot using Streamlit:

```bash
streamlit run app.py
```

This will start the web interface, where you can upload PDFs or images and ask questions based on their content.

## Usage Instructions

### Upload Files

- **PDFs**: Upload your PDF files in the sidebar. The bot will extract and process the text, making it ready for Q&A.
- **Images**: Upload images (JPG, JPEG, PNG) for text analysis.

### Ask Questions

- Type your question in the provided input box and click "Generate Response".
- The chatbot will generate a response based on the content of the uploaded files.

## Language Models (LLMs) Used

The chatbot leverages advanced Language Models (LLMs) to provide accurate and contextually relevant responses:

- **Google Gemini Model**: Utilized for generating detailed responses and analyzing images. The `gemini-pro-vision` model excels in understanding and processing visual and textual data.
- **Google Generative AI Embeddings**: This model is used for creating text embeddings, essential for vector-based search and retrieval using FAISS.
- **ChatGoogleGenerativeAI Model**: Employed to maintain conversational flow and generate answers that consider the context of previous interactions.

These LLMs are integrated via LangChain, which facilitates seamless chaining of models, memory management, and prompt customization.

## File Structure

- `app.py`: The main application file containing the Streamlit interface and chatbot logic.
- `requirements.txt`: Lists all the necessary Python packages.
- `.env`: Stores environment variables like the Google API key (not included in the repository).
- `faiss_index`: The local storage for the FAISS vector store.

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!