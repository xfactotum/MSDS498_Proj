import os
import uuid
import pandas as pd

from models.souschef import Chatbot
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
    for i in range(len(st.session_state.messages) - 1, -1, -1):
        if st.session_state.messages[i]["role"] == "user":
            return st.session_state.messages[i]["content"]


def callback():
    edited_rows = st.session_state["data_editor"]["edited_rows"]
    rows_to_delete = []

    for idx, value in edited_rows.items():
        if value["Select"] is True:
            rows_to_delete.append(idx)

    chopping_mask = st.session_state.data.index.isin(rows_to_delete)
    recipe_ids = st.session_state.data[chopping_mask]["id"].to_list()
    if "key_question" in st.session_state:
        query = st.session_state.key_question
    else:
        query = last_prompt()
    with st.spinner("Processing..."):
        if "selection" in st.session_state:  # if there is another choice selected previously
            st.session_state.chatbot.clear_conversation_history()
        output, tokens_in, tokens_out, t_elapsed = st.session_state.chatbot.answer(query, recipe_ids)
        st.session_state.resp_time = t_elapsed
        st.session_state.tokens_in = st.session_state.tokens_in+ tokens_in
        st.session_state.tokens_out = st.session_state.tokens_out + tokens_out
        #print(f"output: {output}")
        st.session_state.messages.append({"role": "assistant", "content": output})
    st.session_state.selection = recipe_ids

    st.session_state["data"] = (
        st.session_state["data"].drop(st.session_state.data[chopping_mask].index, axis=0).reset_index(drop=True)
    )


def dataframe_with_selections():
    df_with_selections = st.session_state["data"].copy()
    columns = df_with_selections.columns
    if "Select" not in columns:
        df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        key="data_editor",
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True),
                       "name": st.column_config.TextColumn(width="large")
                       },
        disabled=columns,
        on_change=callback
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


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
    st.session_state.chatbot = Chatbot(llm=llm, knowledge_base=vector_db)


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
if "messages" not in st.session_state:  # Initialize the chat messages history
    st.session_state.messages = [{"role": "assistant", "content": "What is in your mind?"}]

# Load recipes
recipe_df = load_data()

if not api_key:
    st.sidebar.warning('API key required to try this app. The API key is not stored in any form.')
else:
    os.environ["OPENAI_API_KEY"] = api_key
    # Initialize model
    if "chatbot" not in st.session_state:
        new_chat(model_name="gpt-4-1106-preview", api_key=api_key)

    # Set session ID
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4()

    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame()

    if "selection" not in st.session_state:
        st.session_state.selection = []

    if "tokens_in" not in st.session_state:
        st.session_state.tokens_in = 0
    if "tokens_out" not in st.session_state:
        st.session_state.tokens_out = 0
    if "resp_time" not in st.session_state:
        st.session_state.resp_time = 0.00

    with st.container(border = True):
        st.caption("FOR DEVELOPMENT PURPOSES ONLY")
        st.markdown("The Chat_1 page will create a dataframe of the 10 most similar retrievals to the "+
                    "User's query. The user will then be asked to select a recipe from this dataframe. "+
                    "Once selected, Sous Chef will generate instructions for this recipe.")
        st.markdown("")
        col1, col2 = st.columns([3,1])
        with col1.container(border = True):
            st.caption("CUMULATIVE FOR THIS SESSION")
            col11, col12, col13 = st.columns(3)
            col11.metric("Prompt Tokens:", st.session_state.tokens_in)
            col12.metric("Response Tokens:", st.session_state.tokens_out)
            col13.metric("API Cost:", "$"+str(round(st.session_state.tokens_in/1000*.01 +
                                               st.session_state.tokens_out/1000*.03,2)))
        col2.metric("Last Response Time:", str(st.session_state.resp_time) + " s")

    st.image("../images/SousChefLogo.png")
    st.markdown("")
    st.markdown("Hello! I'm here to help you decide on a recipe to use while preparing your meal.")
    st.markdown("")

    dataframe_with_selections()

    # Text box for user input
    clear_button = st.button("Reset Conversation", key="clear")

    # Clear conversation when reset button is clicked
    if clear_button:
        new_chat(model_name="gpt-4-1106-preview", api_key=api_key)
        st.session_state.data = pd.DataFrame()
        st.session_state.messages = [{"role": "assistant", "content": "Got it! Let's start over"}]
        st.session_state.messages.append({"role": "assistant", "content": "What is in your mind?"})
        st.session_state.selection = []
        st.session_state.chatbot.clear_conversation_history()
        st.rerun()

    if len(st.session_state.data) == 0 or len(st.session_state.selection) > 0:
        if prompt := st.chat_input("You:"):  # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        if len(st.session_state.data) == 0 and len(st.session_state.selection) == 0:
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    documents = st.session_state.chatbot.get_docs(prompt)
                    st.session_state.key_question = prompt
                    ids = [doc.metadata['recipe_id'] for doc in documents if 'recipe_id' in doc.metadata.keys()]
                    candidates = recipe_df[recipe_df["id"].isin(ids)]
                    if candidates.shape[0] > 0:
                        st.session_state.data = candidates[['name', 'minutes', 'description',
                                                            'tags', 'id']].reset_index(drop=True)
                        message = {"role": "assistant",
                                   "content": "Great! Now choose from the list of recipes above to get "
                                              "step-by-step instructions."
                                   }
                        st.session_state.messages.append(message)
                    else:
                        st.session_state.messages.append({"role": "assistant",
                                                          "content": "Sorry there is not such a recipe in our database."
                                                          "Try again."})
                    st.rerun()
        elif len(st.session_state.selection) > 0:
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response, tokens_in, tokens_out, t_elapsed = st.session_state.chatbot.answer(prompt)
                    st.session_state.resp_time = t_elapsed
                    st.session_state.tokens_in = st.session_state.tokens_in + tokens_in
                    st.session_state.tokens_out = st.session_state.tokens_out + tokens_out
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)
