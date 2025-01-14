import os
import re
import tempfile
import torch
import qdrant_client
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_current_datetime():
    return datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

load_dotenv()
upload_history = []
diffrence = []
client = 0
user_count = get_current_datetime()

def init_vector():

     global client
     client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),)
     vectors_config = qdrant_client.http.models.VectorParams(
            size = 1024,
            distance= qdrant_client.http.models.Distance.COSINE)
     if not client.collection_exists(st.session_state['user_id']):
        client.create_collection(collection_name=st.session_state['user_id'],
            vectors_config=vectors_config)

        return client



def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! How can I help with your documents today? 📄"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey there! 👋"]

    if 'llm' not in st.session_state:
        with st.spinner('Loding Model...'):
            st.session_state['llm'] = load_model()

    if 'user_id' not in st.session_state:
        user_collection = f'user-{user_count}-collection'
        st.session_state['user_id'] = [user_collection]

    if 'uploaded_history' not in st.session_state:
        st.session_state['uploaded_history'] = []

    if 'file_oprations_pending' not in st.session_state:
        st.session_state['file_oprations_pending'] = True


def test_rag(query,qa):
    response = qa.run(query)
    print(f"raw--{response}")
    result = ''
    # Regex pattern to match "Helpful Answer" blocks and stop at the next "Question" or "Unhelpful Answer"
    pattern = re.compile(
        r'(?:Helpful Answer|Unhelpful Answer):\s*(.*?)(?=\n(?:Question:|Unhelpful Answer:|Helpful Answer:)|\Z)',
        re.DOTALL)

    # Initialize the final_result variable
    final_result = ''

    # Process the response
    while True:
        # Find the first match
        match = pattern.search(response)
        if match:
            # Append the result to the final output
            answer = match.group(1).strip()
            final_result += answer + '\n'

            # Update the response by removing the matched part
            response = response[match.end():]

            # Check if the remaining response starts with "Unhelpful Answer" or "Question"
            if response.lstrip().startswith('Unhelpful Answer:') or response.lstrip().startswith('Question:'):
                break
        else:
            break

    # Perform the substitution on the final result
    if final_result:
        pattern = re.compile(r'\$')
        result = pattern.sub(u"\U0001F4B2", final_result)

    return result.strip()


def display_chat(qa):
        reply_container = st.container()
        container = st.container()

        with container:
            if user_input := st.chat_input(placeholder="Ask about your Documents",key='input'):
                with st.spinner('Generating response...'):
                    output = test_rag(user_input, qa)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message_user = st.chat_message("user")
                    message_ai = st.chat_message("assistant")
                    message_user.write(st.session_state["past"][i])
                    message_ai.write(st.session_state["generated"][i])

def load_model():
    # Create llm
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id = "meta-llama/Meta-Llama-3-8B"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cuda",
                                                 quantization_config=nf4_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    query_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_length=1024,
        device_map="auto", )
    llm = HuggingFacePipeline(pipeline=query_pipeline)

    return llm

def create_retrieval_chain(vector_store,llm):
    load_dotenv()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        verbose=True
    )
    return qa


def main():
    # Initialize session state
    load_dotenv()
    initialize_session_state()
    init_vector()

    # Initialize Streamlit
    st.title("WarpLearn 🤖")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            diffrence = list(set([file.name]) - set(st.session_state['uploaded_history']))
            if file.name in diffrence:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)
                elif file_extension == ".docx" or file_extension == ".doc":
                    loader = Docx2txtLoader(temp_file_path)
                elif file_extension == ".txt":
                    loader = TextLoader(temp_file_path)

                if loader:
                    text.extend(loader.load())
                    os.remove(temp_file_path)

            if diffrence:
                st.session_state['file_oprations_pending'] = True
            else:
                st.session_state['file_oprations_pending'] = False

        if st.session_state['file_oprations_pending']:
            with st.spinner('Splitting text...'):
                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=200, length_function=len)
                text_chunks = text_splitter.split_documents(text)

            # Create embeddings
            with st.spinner('creating embeddings...'):
                model_name = "BAAI/bge-large-en"
                model_kwargs = {'device': 'cpu'}
                encode_kwargs = {'normalize_embeddings': False}
                embeddings = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )

            # Create vector store
            with st.spinner('creating vector...'):
                st.session_state['vector_store'] = Qdrant(
                    client=client,
                    collection_name=st.session_state['user_id'],
                    embeddings=embeddings,
                )
                st.session_state['vector_store'].add_documents(text_chunks)

        qa = create_retrieval_chain(st.session_state['vector_store'],st.session_state['llm'])
        st.session_state['uploaded_history'].extend(diffrence)
        display_chat(qa)


if __name__ == "__main__":
    main()