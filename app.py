import streamlit as st
import os
import re
from textblob import TextBlob
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Your Custom Title")

st.markdown(
    """
    <style>
    /* Target user messages based on a data attribute or class (adjust the selector as needed) */
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: flex;
    flex-direction: row-reverse;
    align-itmes: end;
}
    [data-testid="stChatMessageUser"] {
        text-align: right;
    }

    .st-emotion-cache-pkbazv e19011e66{

    }
    </style>
    """,
    unsafe_allow_html=True
)
MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
PINECONE_INDEX_NAME = "simple-rag"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

def correct_spelling(text):
    return str(TextBlob(text).correct())

def clean_question(q):
    q = q.strip()
    return re.sub(r"[\(\[{]$", "", q)

@st.cache_resource
def get_vector_db():
    try:
        # Initialize Pinecone with the new API
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        
        # Check if our index exists
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            st.warning(f"Index '{PINECONE_INDEX_NAME}' not found. Please ingest documents first.")
            return None
        
        # Get the index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Initialize the Pinecone vector store
        return PineconeVectorStore(
            index=index,
            embedding=embedding,
            text_key="text"  # Field containing document text
        )
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {str(e)}")
        return None

def create_retriever(vector_db, llm):
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant helping retrieve city-related documents for a question-answering system.

    Generate 5 different rephrasings of the following question to improve document search and retrieval.

    Keep them semantically similar but vary phrasing, keywords, and structure. Separate each version with a newline.

    Original question: {question}
    """
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # retriever = MultiQueryRetriever.from_llm(
    #     vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    # )
    return retriever

def create_chain(retriever, llm):
    template = """
    You are a helpful, professional and friendly multilingual assistant specialized in answering questions about The city of Arlington, like city council meetings and local matters. If the user asks the question in another language, you have to understand and answer in that same language. They might even tell you the language that they will use. 

    Use the information provided in the context below to answer the question. 

    Be clear and concise, and explain in a way a 15-year-old could understand, unless the user requests a more detailed answer.

    Answer the question even if it includes minor formatting errors, typos, or special characters. You can answer general questions as well.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    st.title("Ask Arlington!")

    # Display the chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
            
    user_input = st.chat_input(placeholder="Enter your question:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        with st.spinner("Generating response"):
            try:
                llm = ChatGoogleGenerativeAI(model=MODEL)

                vector_db = get_vector_db()

                if vector_db is None:
                    st.error("Failed to load the vector db")
                    return
                
                retriever = create_retriever(vector_db, llm)

                chain = create_chain(retriever, llm)

                cleaned_input = clean_question(user_input)
                corrected_input = correct_spelling(cleaned_input)

                response = chain.invoke(corrected_input)

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

            except Exception as e:
                st.error(f"An error has occurred: {str(e)}")

if __name__ == "__main__":
    main()