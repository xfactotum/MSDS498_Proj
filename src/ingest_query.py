from typing import Optional
import os
import chromadb
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb.utils.embedding_functions as embedding_functions

CHROMA_DB_DIRECTORY = '../db'


def ingest_data():
    recipe_df = pd.read_csv("../data/external/RAW_recipes.csv")
    recipe_df["id"] = recipe_df["id"].astype(str)
    recipe_df["description"].fillna("None", inplace=True)
    test_df = recipe_df.iloc[:10, :]
    recipe_collection.add(
        documents=test_df["description"].to_list(),
        metadatas=[{"tags": a} for a in test_df["tags"]],
        ids=test_df["id"].to_list()
    )


def split_texts(load, chunk_size: Optional[int] = 500, chunk_overlap: Optional[int] = 20):
    # instantiate the RecursiveCharacterTextSplitter class
    # by providing the chunk_size and chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Now split the documents into chunks and return
    chunked_docs = splitter.split_text(load)
    return chunked_docs


client = chromadb.PersistentClient(path=CHROMA_DB_DIRECTORY)
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.environ["HUGGINGFACE_API_KEY"],
    model_name="sentence-transformers/all-mpnet-base-v2"
)
recipe_collection = client.get_or_create_collection(name="hf_embeddings", embedding_function=huggingface_ef)
# ingest_data()
query = "Give me a spicy recipe"
result = recipe_collection.query(query_texts=[query], n_results=5, include=["documents", 'distances',])
print("Query:", query)
print("Most similar recipes:")
# Extract the first (and only) list inside 'ids'
ids = result.get('ids')[0]
# Extract the first (and only) list inside 'documents'
documents = result.get('documents')[0]
# Extract the first (and only) list inside 'documents'
distances = result.get('distances')[0]

for id_, document, distance in zip(ids, documents, distances):
    # Cosine Similiarity is calculated as 1 - Cosine Distance
    print(f"ID: {id_}, Recipe: {document}, Similarity: {1 - distance}")

