# arch.py

import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)


class HybridArchitecture:
    """
    Implements a hybrid RAG architecture that:
    1. Generates an answer from a private, local knowledge base.
    2. Generates an answer from a live web search.
    3. Uses a "judge" LLM to select the better of the two answers.
    """

    def __init__(self, model_name="google/flan-t5-base"):
        print("Initializing the architecture...")
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text2text-generation",
            model_kwargs={"temperature": 0.1, "max_length": 512, "dtype": torch.bfloat16},
        )
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
        answer_chain = LLMChain(llm=self.llm, prompt=prompt)
        response = answer_chain.invoke({"context": context, "query": query})
        answer = response['text'].strip()
        print(f"   Local Answer: {answer}")
        return answer

    def generate_from_web_search(self, query: str) -> str:
        print("-> Generating answer from Web Search...")
        context = self.web_retriever.run(query)
        
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="Based ONLY on the context below, answer the question. If you cannot, say 'Answer not found'.\nContext: {context}\nQuestion: {query}\nAnswer:"
        )
        answer_chain = LLMChain(llm=self.llm, prompt=prompt)
        response = answer_chain.invoke({"context": context, "query": query})
        answer = response['text'].strip()
        print(f"   Web Answer: {answer}")
        return answer

    def choose_best_answer(self, query: str, answer_local: str, answer_web: str) -> str:
        print("\n-> The Judge is choosing the best answer...")
        prompt = PromptTemplate(
            input_variables=["query", "answer_local", "answer_web"],
            template="""
            You are a judge. Your task is to determine which of the two answers below is a better response to the User Query.
            Consider completeness, accuracy, and relevance.
            
            User Query: "{query}"
            
            Answer LOCAL: "{answer_local}"
            Answer WEB: "{answer_web}"
            
            Which answer is better? You must choose one. Respond only with the word LOCAL or WEB.
            Decision:
            """
        )
        
        judge_chain = LLMChain(llm=self.llm, prompt=prompt)
        response = judge_chain.invoke({
            "query": query, "answer_local": answer_local, "answer_web": answer_web
        })
        
        decision = response['text'].strip().upper()
        print(f"   Judge's Decision: {decision}")
        return decision

    def run(self, query: str):
        answer_local = self.generate_from_local_kb(query)
        answer_web = self.generate_from_web_search(query)

        is_local_valid = "answer not found" not in answer_local.lower()
        is_web_valid = "answer not found" not in answer_web.lower()

        if is_local_valid and not is_web_valid:
            print("\n--- Final Result --- \n✅ Web search failed, using valid Local KB answer.")
            return {"final_answer": answer_local, "source": "Local KB"}

        if not is_local_valid and is_web_valid:
            print("\n--- Final Result --- \n✅ Local KB failed, using valid Web Search answer.")
            return {"final_answer": answer_web, "source": "Web Search"}

        if not is_local_valid and not is_web_valid:
            print("\n--- Final Result --- \n❌ Both sources failed to find an answer.")
            return {"final_answer": "Could not find an answer from any source.", "source": "None"}

        # If both are valid, let the judge decide
        decision = self.choose_best_answer(query, answer_local, answer_web)
        
        print("\n--- Final Result ---")
        if "WEB" in decision:
            print("✅ Judge selected the Web Search answer.")
            return {"final_answer": answer_web, "source": "Web Search (Chosen by Judge)"}
        else:
            print("✅ Judge selected the Local KB answer.")
            return {"final_answer": answer_local, "source": "Local KB (Chosen by Judge)"}