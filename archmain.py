# archmain.py

from arch import HybridArchitecture
from langchain_core.documents import Document  # <-- THIS LINE IS FIXED
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

if __name__ == "__main__":
    
    pipeline = HybridArchitecture()

    # Define our private, local knowledge base
    local_knowledge_docs = [
        Document(page_content="ayush i my name . i have blue pen "),
        Document(page_content="prakash is my DSA tecaher and Mini project head ")
    ]
    
    # Load the private documents into the pipeline
    pipeline.setup_knowledge_base(local_knowledge_docs)

    # --- Test Case 1: A question best answered by our LOCAL knowledge base ---
    user_query_1 = "what is meta ?"
    print(f"\n================================\nExecuting pipeline for query: '{user_query_1}'\n================================")
    result_1 = pipeline.run(user_query_1)
    
    print("\n\n--- PIPELINE OUTPUT ---")
    print(f"Final Answer: {result_1['final_answer']}")
    print(f"Source: {result_1['source']}")
    print("-----------------------\n\n")

    # --- Test Case 2: A question that can ONLY be answered by a WEB search ---
    user_query_2 = "Who is Prakash Hegade "
    print(f"\n================================\nExecuting pipeline for query: '{user_query_2}'\n================================")
    result_2 = pipeline.run(user_query_2)
    
    print("\n\n--- PIPELINE OUTPUT ---")
    print(f"Final Answer: {result_2['final_answer']}")
    print(f"Source: {result_2['source']}")
    print("-----------------------")