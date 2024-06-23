import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts=[
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
def get_gemini_response(input,image,prompt):
    try:
        model= genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([input,image[0],prompt])
        return response.text
    except:
        return 'Trouble encountered while generating response'

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer
    {context}

    {chat_history}
    human_input: {human_input}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
    
    prompt = PromptTemplate(input_variables=["context","chat_history", "human_input"], template=prompt_template)

    #prompt = PromptTemplate(template = prompt_template, input_variables = ["context","chat_history", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history",input_key="human_input")

    chain = load_qa_chain(model, chain_type="stuff",memory=memory, prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "human_input": user_question}
        , return_only_outputs=True)
    
    text_chunks = get_text_chunks(chain.memory.buffer)
    get_vector_store(text_chunks)

    print(response)
    st.subheader("The Response is")
    st.write("", response["output_text"])
    
    #st.write(response)



def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Custom chatbot for pdf / image analysis")

    with st.sidebar:
        st.title("Get Started:")
        uploaded_file = st.file_uploader("Upload your Files and Click on the Process Button.", accept_multiple_files=True, type=["jpg", "jpeg", "png", "pdf"])
        file_extension = ''
        file_name = ''
        if uploaded_file is not None:
            for file in uploaded_file:
                file_extension = file.name.split('.')[-1]
                file_name = file.name
                st.write(f"Uploaded File Format: {file_extension}")

        # if st.button("Process"):
            if file_extension == 'pdf':
                with st.spinner("We are getting prepared for you..."):
                    raw_text = get_pdf_text(uploaded_file)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            elif file_extension in ['png', 'jpg', 'jpeg']:
                with st.spinner("Processing..."):
                    st.success("Done")

    if (file_extension in ['png', 'jpg', 'jpeg']):
        image = Image.open(uploaded_file[0])
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.form("question_form"):
        user_question = st.text_input("Ask a Question")
        submit_button = st.form_submit_button(label="Generate Response")

    if submit_button:
        if file_extension == 'pdf':
            user_input(user_question)
        
        if (file_extension in ['png', 'jpg', 'jpeg']):
            input_prompt = """ 
            You are an expert in reading textual content from images. We will upload an image containing text data. 
            Your task is to extract relevant information from the image and provide responses to questions based on it.

            For example, if the image contains a story about "Jack and the Beanstalk," and the text mentions that "Jack had 3 beans," 
            then when asked "How many beans did Jack have?" you should respond with "Jack had 3 beans."

            Ensure your responses are accurate and relevant to the content in the image.
            """
            image_data=input_image_setup(uploaded_file[0])
            response=get_gemini_response(input_prompt,image_data,user_question)
            st.subheader("The Response is")
            st.write(response)


if __name__ == "__main__":
    main()
