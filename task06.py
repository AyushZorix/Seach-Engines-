import requests 
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
import torch
from transformers import BertModel, BertTokenizer
from tqdm.auto import tqdm

# Data loading
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# Create DataFrame
df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])

# Example documents for basic vectorization
docs_example = [
    "January course details, register now",
    "Course prerequisites listed in January catalog",
    "Submit January course homework by end of month",
    "Register for January course, no prerequisites",
    "January course setup: Python and Google Cloud"
]

# Basic CountVectorizer example
cv = CountVectorizer()
cv.fit(docs_example)
names = cv.get_feature_names_out()
X = cv.transform(docs_example)

# Create DataFrame for visualization
df_docs = pd.DataFrame(X.toarray(), columns=names).T
print("Basic CountVectorizer results:")
print(df_docs)

# CountVectorizer with stop words
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(docs_example)
names = cv.get_feature_names_out()
df_docs = pd.DataFrame(X.toarray(), columns=names).T
print("\nCountVectorizer with stop words:")
print(df_docs)

# TF-IDF Vectorizer
cv = TfidfVectorizer(stop_words='english')
X = cv.fit_transform(docs_example)
names = cv.get_feature_names_out()
df_docs = pd.DataFrame(X.toarray(), columns=names).T
print("\nTF-IDF Vectorizer results:")
print(df_docs.round(2))

# Query example
query = "Do I need to know python to sign up for the January course?"
q = cv.transform([query])
query_dict = dict(zip(names, q.toarray()[0]))
print(f"\nQuery vector: {query_dict}")

# Cosine similarity
cosine_scores = cosine_similarity(X, q)
print(f"\nCosine similarity scores: {cosine_scores.flatten()}")

# TextSearch class implementation
class TextSearch:
    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')

# Initialize and use TextSearch
fields = ['section', 'question', 'text']
transformers = {}
matrices = {}

for field in fields:
    cv = TfidfVectorizer(stop_words='english', min_df=3)
    X = cv.fit_transform(df[field])
    transformers[field] = cv
    matrices[field] = X

# Create TextSearch index
index = TextSearch(text_fields=['section', 'question', 'text'])
index.fit(documents)

# Search example
query = "I just signed up. Is it too late to join the course?"
results = index.search(
    query=query,
    n_results=5,
    boost={'question': 3.0},
    filters={'course': 'data-engineering-zoomcamp'}
)

print(f"\nSearch results for: '{query}'")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['question']}")
    print(f"   {result['text'][:100]}...")

# SVD example
X = matrices['text']
cv = transformers['text']

svd = TruncatedSVD(n_components=16)
X_emb = svd.fit_transform(X)

Q = cv.transform([query])
Q_emb = svd.transform(Q)

score = cosine_similarity(X_emb, Q_emb).flatten()
idx = np.argsort(-score)[:5]

print(f"\nSVD-based search results:")
for i, doc_idx in enumerate(idx, 1):
    print(f"{i}. {df.iloc[doc_idx]['text'][:100]}...")

# NMF example
nmf = NMF(n_components=16)
X_emb_nmf = nmf.fit_transform(X)
Q_emb_nmf = nmf.transform(Q)

score_nmf = cosine_similarity(X_emb_nmf, Q_emb_nmf).flatten()
idx_nmf = np.argsort(-score_nmf)[:5]

print(f"\nNMF-based search results:")
for i, doc_idx in enumerate(idx_nmf, 1):
    print(f"{i}. {df.iloc[doc_idx]['text'][:100]}...")

# BERT embeddings functions
def make_batches(seq, n):
    result = []
    for i in range(0, len(seq), n):
        batch = seq[i:i+n]
        result.append(batch)
    return result

def compute_embeddings(texts, batch_size=8):
    """Compute BERT embeddings for a list of texts"""
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    
    text_batches = make_batches(texts, batch_size)
    
    all_embeddings = []
    
    for batch in tqdm(text_batches):
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    
        with torch.no_grad():
            outputs = model(**encoded_input)
            hidden_states = outputs.last_hidden_state
            
            batch_embeddings = hidden_states.mean(dim=1)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_np)
    
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings

# Example: Compute embeddings for text field (commented out as it's computationally intensive)
"""
print("Computing BERT embeddings...")
embeddings = {}
fields = ['section', 'question', 'text']

for f in fields:
    print(f'Computing embeddings for {f}...')
    embeddings[f] = compute_embeddings(df[f].tolist())

# Save embeddings
with open('embeddings.bin', 'wb') as f_out:
    pickle.dump(embeddings, f_out)

print("Embeddings saved to embeddings.bin")
"""

print("\nCode execution completed!")