# arch.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import warnings

warnings.filterwarnings("ignore")

class HybridArchitecture:
    def __init__(self):
        print("Initializing the architecture with Gemini Pro...")
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        self.web_retriever = DuckDuckGoSearchRun()
        self.local_retriever = None
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Initialization complete.")

    def setup_knowledge_base(self, documents):
        print("Setting up the local knowledge base...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        db = FAISS.from_documents(docs, self.embedding_model)
        self.local_retriever = db.as_retriever(search_kwargs={"k": 2})
        print("Local knowledge base is ready.")

    def generate_from_local_kb(self, query: str) -> str:
        print("\n-> Generating answer from Local Knowledge Base...")
        if not self.local_retriever:
            return "Local knowledge base is not configured."
        retrieved_docs = self.local_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="Based ONLY on the context below, answer the question. If you cannot, say 'Answer not found'.\nContext: {context}\nQuestion: {query}\nAnswer:"
        )
        answer_chain = prompt | self.llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "query": query})
        print(f"   Local Answer: {answer}")
        return answer

    def generate_from_web_search(self, query: str) -> str:
        print("-> Generating answer from Web Search...")
        context = self.web_retriever.run(query)
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="Based ONLY on the context below, answer the question. If you cannot, say 'Answer not found'.\nContext: {context}\nQuestion: {query}\nAnswer:"
        )
        answer_chain = prompt | self.llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "query": query})
        print(f"   Web Answer: {answer}")
        return answer

    def choose_best_answer(self, query: str, answer_local: str, answer_web: str) -> str:
        print("\n-> The Judge (Gemini) is choosing the best answer...")
        prompt = PromptTemplate(
            input_variables=["query", "answer_local", "answer_web"],
            template="""
            You are a judge... Respond only with the word LOCAL or WEB.
            Decision:
            """
        )
        judge_chain = prompt | self.llm | StrOutputParser()
        decision = judge_chain.invoke({
            "query": query, "answer_local": answer_local, "answer_web": answer_web
        }).strip().upper()
        print(f"   Judge's Decision: {decision}")
        return decision

    def run(self, query: str):
        answer_local = self.generate_from_local_kb(query)
        answer_web = self.generate_from_web_search(query)
        is_local_valid = "answer not found" not in answer_local.lower()
        is_web_valid = "answer not found" not in answer_web.lower()

        if is_local_valid and not is_web_valid:
            return {"final_answer": answer_local, "source": "Local KB"}
        if not is_local_valid and is_web_valid:
            return {"final_answer": answer_web, "source": "Web Search"}
        if not is_local_valid and not is_web_valid:
            return {"final_answer": "Could not find an answer from any source.", "source": "None"}
        
        decision = self.choose_best_answer(query, answer_local, answer_web)
        
        if "WEB" in decision:
            return {"final_answer": answer_web, "source": "Web Search (Chosen by Judge)"}
        else:
            return {"final_answer": answer_local, "source": "Local KB (Chosen by Judge)"}