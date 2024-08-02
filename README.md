To install and run your Retrieval Augmented Generation (RAG) chatbot, follow these steps:

### Installation Guide

1. **Create a `.env` File**:
   - In the root directory of your project, create a `.env` file.
   - Add your Google API key in the following format:
     ```
     GOOGLE_API_KEY=<your_key>
     ```

2. **Set Up a Virtual Environment**:
   - It is recommended to create and use a separate virtual environment to avoid conflicts with other packages. If you are using `conda`, you can create an environment by following the [getting started guide](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

3. **Install Required Packages**:
   - Once your virtual environment is activated, install the necessary packages by running the following command:
     ```bash
     pip install -r requirements.txt
     ```

### How to Run the Chatbot

After successfully installing all the dependencies, you can run the chatbot using Streamlit:

```bash
streamlit run app.py
```

This command will start the Streamlit app, and you can interact with your RAG chatbot through the web interface.

Let me know if you need any further assistance!