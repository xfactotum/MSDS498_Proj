from dotenv import load_dotenv, find_dotenv
import pandas as pd

# import all langchain modules
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# import all the chromadb modules
import chromadb


def ingest_data():
    recipe_df = pd.read_parquet("../data/processed/final_recipe_sample.parquet")
    recipe_df["id"] = recipe_df["id"].astype(str)

    test_df = recipe_df.iloc[:, :]

    test_df["detailed_recipe"] = ("name:'" + test_df["name"] + "'\ndescription:'" +
                                  test_df["description"] + "'\ninstructions:" +
                                  test_df["steps"])

    metadata_lst = []
    for i, row in test_df[["ingredients", "minutes", "tags", "id"]].iterrows():
        print(f"Processing item: {i}...")
        # export each ingredient and tag as a searchable key
        metadata = {f"{i}": "ing" for i in eval(row["ingredients"])}
        metadata.update({f"{i}": "tag" for i in eval(row["tags"])})
        metadata['minutes'] = row["minutes"]
        metadata['recipe_id'] = row["id"]
        metadata_lst.append(metadata)

    print(f"Inserting {test_df.shape[0]} items into the db...")
    vector_db.add_texts(
        texts=test_df["detailed_recipe"].to_list(),
        metadatas=metadata_lst,
        ids=test_df["id"].to_list()
    )


# Load environment variables
load_dotenv(find_dotenv())

CHROMA_DB_DIRECTORY = '../db'

client = chromadb.PersistentClient(path=CHROMA_DB_DIRECTORY)
embeddings = OpenAIEmbeddings()
vector_db = Chroma(
    client=client,
    embedding_function=embeddings
)

ingest_data()
