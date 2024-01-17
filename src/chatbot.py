import os
import pandas as pd

from models.gpt4all_model import MyGPT4ALL

# import all langchain modules
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# import all the chromadb modules
import chromadb


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
# GPT4ALL_MODEL_NAME = 'gpt4all-13b-snoozy-q4_0.gguf'
GPT4ALL_MODEL_NAME = 'nous-hermes-llama2-13b.Q4_0.gguf'
GPT4ALL_MODEL_FOLDER_PATH = f'{PROJECT_DIRECTORY}/models'
GPT4ALL_ALLOW_STREAMING = True
GPT4ALL_ALLOW_DOWNLOAD = False

client = chromadb.PersistentClient(path=CHROMA_DB_DIRECTORY)
embeddings = GPT4AllEmbeddings()
vector_db = Chroma(
    client=client,
    embedding_function=embeddings
)
# ingest_data()
llm = MyGPT4ALL(
    model_folder_path=GPT4ALL_MODEL_FOLDER_PATH,
    model_name=GPT4ALL_MODEL_NAME,
    allow_streaming=GPT4ALL_ALLOW_STREAMING,
    allow_download=GPT4ALL_ALLOW_DOWNLOAD
)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True, verbose=True
)
while True:
    # query = "Give me a spicy recipe"
    query = input("What's on your mind: ")
    if query == 'exit':
        break
    result = qa_chain(query)
    answer, docs = result['result'], result['source_documents']

    print(answer)

    print("#" * 30, "Recipes", "#" * 30)
    for document in docs:
        print("\n> TAGS: " + document.metadata["tags"] + ":")
        print(document.page_content)
    print("#" * 30, "Recipes", "#" * 30)
