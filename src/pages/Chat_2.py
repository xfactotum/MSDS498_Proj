import os
import uuid
import pandas as pd

from models.souschef2 import Chatbot
from dotenv import load_dotenv, find_dotenv

# import all langchain modules
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# import all the chromadb modules
import chromadb

# import streamlit modules
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_data():
    df = pd.read_parquet("../data/processed/final_recipe_sample.parquet")
    df["id"] = df["id"].astype(str)
    return df


def last_prompt():
    for i in range(len(st.session_state.messages2) - 1, -1, -1):
        if st.session_state.messages2[i]["role"] == "user":
            return st.session_state.messages2[i]["content"]


def new_chat(model_name: str, api_key: str):
    client = chromadb.PersistentClient(path=CHROMA_DB_DIRECTORY)
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(
        client=client,
        embedding_function=embeddings
    )

    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key
    )
    # Persist our chatbot instance
    st.session_state.chatbot2 = Chatbot(llm=llm, knowledge_base=vector_db)


# Load environment variables
load_dotenv(find_dotenv())
# Set vector db location
CHROMA_DB_DIRECTORY = '../db'

# Configure streamlit app title, icon, and layout
st.set_page_config(
    page_title="Sous Chef Chatbot",
    page_icon=":robot_face:",
    layout="wide"
)

# Prompt user for the API key if it is not already set in an environment variable
if "OPENAI_API_KEY" in os.environ.keys():
    api_key = os.environ["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("API KEY", type="password")

# Initialize session states
if "messages2" not in st.session_state:  # Initialize the chat messages history
    st.session_state.messages2 = [{"role": "assistant", "content": "What is in your mind?"}]

# Load recipes
recipe_df = load_data()

if not api_key:
    st.sidebar.warning('API key required to try this app. The API key is not stored in any form.')
else:
    os.environ["OPENAI_API_KEY"] = api_key
    # Initialize model
    if "chatbot2" not in st.session_state:
        new_chat(model_name="gpt-4-1106-preview", api_key=api_key)

    # Set session ID
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4()

    if "data2" not in st.session_state:
        st.session_state.data2 = pd.DataFrame()

    if "selection2" not in st.session_state:
        st.session_state.selection2 = []

    if "tokens_in2" not in st.session_state:
        st.session_state.tokens_in2 = 0
    if "tokens_out2" not in st.session_state:
        st.session_state.tokens_out2 = 0
    if "resp_time2" not in st.session_state:
        st.session_state.resp_time2 = 0.00
    if "k" not in st.session_state:
        st.session_state.k = 3

    with ((((st.container(border = True))))):
        st.caption("FOR DEVELOPMENT PURPOSES ONLY")
        st.markdown("The Chat_2 page will retrieve the k most similar recipes to the User's query. " +
                    "These k recipes are used as context when generating the response.")
        st.markdown("")
        colA, colB = st.columns([2, 1])
        with colA:
            st.session_state.k = st.select_slider('Select a value for k:', options = [3,4,5,6,7,8,9,10],value = 3)
        col1, col2 = st.columns([3, 1])
        with col1.container(border=True):
            st.caption("CUMULATIVE FOR THIS SESSION")
            col11, col12, col13 = st.columns(3)
            col11.metric("Prompt Tokens:", st.session_state.tokens_in2)
            col12.metric("Response Tokens:", st.session_state.tokens_out2)
            col13.metric("API Cost:", "$" + str(round(st.session_state.tokens_in2 / 1000 * .01 +
                                                      st.session_state.tokens_out2 / 1000 * .03, 2)))
        col2.metric("Last Response Time:", str(st.session_state.resp_time2) + " s")

    st.image("../images/SousChefLogo.png")
    st.markdown("")
    st.markdown("Hello! I'm here to help you decide on a recipe to use while preparing your meal.")
    st.markdown("")

    # Text box for user input
    clear_button = st.button("Reset Conversation", key="clear")

    # Clear conversation when reset button is clicked
    if clear_button:
        new_chat(model_name="gpt-4-1106-preview", api_key=api_key)
        st.session_state.data2 = pd.DataFrame()
        st.session_state.messages2 = [{"role": "assistant", "content": "Got it! Let's start over"}]
        st.session_state.messages2.append({"role": "assistant", "content": "What is in your mind?"})
        st.session_state.selection2 = []
        st.session_state.chatbot2.clear_conversation_history()
        st.rerun()

    if len(st.session_state.data2) == 0 or len(st.session_state.selection2) > 0:
        if prompt := st.chat_input("You:"):  # Prompt for user input and save to chat history
            st.session_state.messages2.append({"role": "user", "content": prompt})

    for message in st.session_state.messages2:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages2[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            st.session_state.key_question2 = prompt
            if "key_question2" in st.session_state:
                query = st.session_state.key_question2
            else:
                query = last_prompt()
            with st.spinner("Processing..."):
                output, tokens_in, tokens_out, t_elapsed = \
                    st.session_state.chatbot2._call(query, k = st.session_state.k)
                st.session_state.resp_time2 = t_elapsed
                st.session_state.tokens_in2 = st.session_state.tokens_in2 + tokens_in
                st.session_state.tokens_out2 = st.session_state.tokens_out2 + tokens_out
                st.session_state.messages2.append({"role": "assistant", "content": output})
            st.rerun()
