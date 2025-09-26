import jwt
import time
import requests
import tiktoken
from langchain_google_vertexai import VertexAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain_google_vertexai import VertexAIEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()
github_token = os.getenv("GITHUB_APP_ID")
github_private_file = os.getenv("GITHUB_PRIVATE_KEY")

from langchain_text_splitters import (
   Language,
   RecursiveCharacterTextSplitter,
)
import vertexai


cached_jwt = None
jwt_expiry = 0  
def get_jwt():
       global cached_jwt, jwt_expiry
       now = int(time.time())


       with open(github_private_file, "r") as f:
           private_key = f.read()
      
       # Check if JWT is still valid
       if cached_jwt and now < jwt_expiry - 60:  # Refresh 1 min before expiry
           return cached_jwt  # Return cached JWT


       # Generate a new JWT
       payload = {
           "iat": now,
           "exp": now + 600,  # Expires in 10 min
           "iss": github_token,  # GitHub App ID
       }
      
       cached_jwt = jwt.encode(payload, private_key, algorithm="RS256")
       jwt_expiry = now + 600  # Update expiry time
       # st.write(f"Now: {now}, Expiration Time: {now + 600}")


       return cached_jwt  # Return new JWT


def get_installation_token():
       jwt_token = get_jwt()
       headers = {
           "Authorization": f"Bearer {jwt_token}",
           "Accept": "application/vnd.github+json",
       }


       # Fetch installation ID
       url = "https://api.github.com/app/installations"
       response = requests.get(url, headers=headers)




       installations = response.json()
       installation_id = installations[0]["id"]


       # Generate installation token
       token_url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
       token_response = requests.post(token_url, headers=headers).json()


       return token_response["token"]  # This lasts 1 hour
  


def count_tokens(text, model_name="text-embedding-004"):
   enc = tiktoken.get_encoding("cl100k_base")  # Good approximation
   return len(enc.encode(text))


def get_text_chunks(extracted_text,max_tokens = 20000, overlap = 200):
   java_splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.JAVA, chunk_size=1500, chunk_overlap=50
   )


   if isinstance(extracted_text, list):
       extracted_text = " ".join(extracted_text)
      
   words = java_splitter.split_text(extracted_text)
  
   chunks = []
   start = 0


   while start < len(words):
       end = min(start + (max_tokens // 1500), len(words))
       chunk = " ".join(words[start:end])
      
       # Ensure the chunk size is within the limit
       while count_tokens(chunk) > max_tokens and end > start:
           end -= 100  # Reduce chunk size if too large
           chunk = " ".join(words[start:end])
      
       chunks.append(chunk)
       start = end - (overlap // 1500)  # Ensure overlap for continuity


   return chunks

def get_vector_store(text_chunks):
   chunks = get_text_chunks(text_chunks)
   PROJECT_ID = "langchain-iris-chatbot"
   #PROJECT_ID = "coderefactoringai"
   LOCATION = "us-central1"
   vertexai.init(project=PROJECT_ID, location=LOCATION)
   model = VertexAIEmbeddings(model_name="text-embedding-004")


   # Wrap chunks into LangChain Document objects
   documents = []
   for chunk in chunks:
       token_count = count_tokens(chunk)
       print(f"Chunk token count: {token_count}")
       documents.append(Document(page_content=chunk))


   # Create FAISS vector store
   vector_store = FAISS.from_documents(documents, model)
   return vector_store


def get_conversation_chain(vector_store):
   retriever = vector_store.as_retriever(
       search_type = "mmr",
       search_kwargs = {"k":8},
   )


   llm = VertexAI(
       project = "langchain-iris-chatbot",
       location="us-central1",
       model = "gemini-2.5-pro",
       model_kwargs={
           "temperature": 0.7,
           "max_length": 600,
           "top_p": 0.95,
           "top_k": 3,
       },
   )
   memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
   conversation_chain = ConversationalRetrievalChain.from_llm(
       llm=llm,
       retriever=retriever,
       memory=memory,
   )
   return conversation_chain
