import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
import os
import getpass

genai.configure(api_key='PUT API KEY HERE')
model =genai.GenerativeModel('gemini-1.5-flash')

doc= PyPDFLoader("../../sem7/major project/MAIN_PROJECT_REPORT_2024_80-123-112_final.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(doc)
texts= [doc.page_content for doc in docs]


class SentenceTransformerWrapper:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text]).tolist()[0]


embedding_model  = SentenceTransformerWrapper('all-MiniLM-L6-v2')



vectorStore = Chroma.from_documents(documents=docs,embedding=embedding_model )


class GoogleGenerativeRunnable(Runnable):
    def __init__(self, model):
        self.model = model

    def invoke(self, inputs, config=None) -> str:
        if isinstance(inputs, str):
            prompt = inputs
        elif hasattr(inputs, "to_string"):
            prompt = inputs.to_string()
        else:
            raise TypeError(f"Unexpected input type: {type(inputs)}")

        response = self.model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text


model_runnable = GoogleGenerativeRunnable(model)

retriever = vectorStore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)





rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model_runnable
    | StrOutputParser()
)

response = rag_chain.invoke({"question": "what makes this project more useful?"})
print(response)




