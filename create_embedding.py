import pickle   #  pip install pickle 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 

with open("chunks.pkl","rb") as file:  #rb =read in binary
    chunks=pickle.load(file)
#  pickle is formate where our data is stored in binary formate and we can read it using pickle.load() method and write it using pickle.dump() method   


#embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    )

#create Vector store
vector_store =Chroma.from_texts(chunks,embeddings,persist_directory='./chroma_db',embedding_function=embeddings)

while True:
    query =input("Enter your query :")
    if query.lower() =="exit":
        break

    results=vector_store.similarity_search(query,k=5) 

    for i ,index in enumerate(results):

      print(f"Result {i+1} :\n{index.page_content}\n")
    