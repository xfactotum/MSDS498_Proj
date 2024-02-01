import os
import pandas as pd

from langchain.chat_models import ChatOpenAI
# import all langchain modules

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings

# import all the chromadb modules
import chromadb


'''
DEPENDENCIES FOR STREAMLIT
'''
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *


'''
question:
How do we get ingest_data() to pick up Mike C's parquet file? Also, does parquet need to be added in order into db -> docs -> Makefile?
'''

def ingest_data():
    recipe_df = pd.read_csv(f'{PROJECT_DIRECTORY}/data/external/RAW_recipes.csv')
    recipe_df["id"] = recipe_df["id"].astype(str)
    recipe_df["description"].fillna("None", inplace=True)
    test_df = recipe_df.iloc[:10000, :]
    vector_db.add_texts(
        texts=test_df["description"].to_list(),
        metadatas=[{"tags": a} for a in test_df["tags"]],
        ids=test_df["id"].to_list()
    )


MODULE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIRECTORY = os.path.dirname(MODULE_DIRECTORY)
CHROMA_DB_DIRECTORY = f'{PROJECT_DIRECTORY}/db'


client = chromadb.PersistentClient(path=CHROMA_DB_DIRECTORY)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    client=client,
    embedding_function=embeddings
)
# ingest_data()

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key="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")



'''
look more into ChromaDB parameters & similarity searches
'''
retriever = vector_db.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True, verbose=True
)


st.subheader("Chatbot with Langchain, ChatGPT, ChromaDB, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi, my name is SousChef! You can ask me recipe and food prep related questions like 'Make me a spicy recipe'! How may I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
#prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name=vector_db), human_msg_template])
                                                    
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


# Set the title of your Streamlit app
st.title("SousChefAI")

# Use HTML to set the favicon to a chef emoji
st.markdown('<link rel="shortcut icon" href="https://cdn.create.vista.com/api/media/medium/32224987/stock-vector-chef-cartoon-with-thumb-up?token=," type="image/x-icon">', unsafe_allow_html=True)
st.markdown('<link rel="icon" href="data:https://cdn.create.vista.com/api/media/medium/32224987/stock-vector-chef-cartoon-with-thumb-up?token=;," type="image/x-icon">', unsafe_allow_html=True)

# Your Streamlit app content
st.write("Hello, this is your SousChefAI a chat assistant for recipe development!")


...
response_container = st.container()
textcontainer = st.container()
...
with textcontainer:
    query = st.text_input("Query: ", key="input")
    ...
with response_container:
    if st.session_state['responses']:   
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


...

'''
error possibly here w. vector_db variable assignment
'''
if query:
    with st.spinner("typing..."):
        ...
        response = conversation.predict(input=f"Context:\n {qa_chain} \n\n Query:\n{query}")
    st.session_state.requests.append(query)
    st.session_state.responses.append(response)

def query_refiner(conversation, query):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']


'''
error possibly here
'''

def find_match(input):
    # Encode the input into a vector
    input_em = model.encode(input).tolist()

    # Query ChromaDB with the encoded vector
    # Replace 'your_index_name' with the name of your ChromaDB index
    results = client.vector_search(index=docs, vector=input_em, top_k=2)

    # Extracting and formatting the results
    first_match = results['matches'][0]['metadata']['text']
    second_match = results['matches'][1]['metadata']['text']

    return first_match + "\n" + second_match