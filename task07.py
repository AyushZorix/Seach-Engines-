import re
import math
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


#This defines normalization rules for product descriptions:
#"12 v" → "12v"
#"1.5 kg" → "1500g"

UNIT_PATTERNS = [
    (re.compile(r'\b(\d+)\s*v\b', flags=re.I), r'\1v'),
    (re.compile(r'\b(\d+(\.\d+)?)\s*kg\b', flags=re.I),
     lambda m: f"{float(m.group(1))*1000:.0f}g"),
    (re.compile(r'\s+'), ' ')
]



#Ensures every value (product attribute) is converted into a clean lowercase string
#Applies the above unit patterns
#Example: " 1.5 Kg " → "1500g"
def normalize_value(val: str) -> str:
    if not isinstance(val, str):
        val = str(val)
    s = val.strip()
    for pat, repl in UNIT_PATTERNS:
        s = pat.sub(repl, s)
    return s.lower()


    
# join all attributes into a single string
def flatten_prod(rec: dict) -> str:
    parts = []
    for k, v in rec.items():
        if k == "id":
            continue
        nk = k.strip().lower().replace(" ", "_")
        nv = normalize_value(v)
        parts.append(f"{nk}: {nv}")
    return " | ".join(parts)



# Weak Ties Recommender

class WeakTiesRecommender:
    def __init__(self):
        self.df = None
        self.tfidf = None
        self.X_tfidf = None
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.svd = None
        self.prod_embeddings = None
        self.G = None
        self.product_nodes = []
        self.attribute_nodes = set()
        self.attr_df_count = {}
        self.N = 0

    def fit(self, records):
        """Build index, graph, and embeddings from product records (list of dicts)."""
        rows = []
        for p in records:
            rows.append({"id": str(p["id"]), "text": flatten_prod(p), **p})
        self.df = pd.DataFrame(rows).set_index("id", drop=False)

        # TF–IDF
        self.tfidf = TfidfVectorizer(min_df=1, stop_words="english")
        self.X_tfidf = self.tfidf.fit_transform(self.df["text"])

        # SVD embeddings
        self.svd = TruncatedSVD(
            n_components=min(50, self.X_tfidf.shape[1]-1 or 1), random_state=42)
        self.prod_embeddings = self.svd.fit_transform(self.X_tfidf)

        # Index maps
        self.id_to_idx = {str(i): j for j, i in enumerate(self.df.index.astype(str))}
        self.idx_to_id = {v: k for k, v in self.id_to_idx.items()}
        self.N = len(self.df)

        # Build bipartite graph
        self.G = nx.Graph()
        self.product_nodes = []
        self.attribute_nodes = set()
        for idx, row in self.df.iterrows():
            pid = f"p_{idx}"
            self.product_nodes.append(pid)
            self.G.add_node(pid, bipartite="product", pid=idx)
            tokens = [t.strip() for t in row["text"].split("|")]
            for t in tokens:
                if not t:
                    continue
                a_node = f"a_{t}"
                self.attribute_nodes.add(a_node)
                self.G.add_node(a_node, bipartite="attr", attr=t)
                self.G.add_edge(pid, a_node, weight=1.0)
        self.attr_df_count = {a: self.G.degree[a] for a in self.attribute_nodes}

    def _normalize(self, v):
        v = np.array(v)
        mn, mx = v.min(), v.max()
        if mx <= mn:
            return np.zeros_like(v)
        return (v - mn) / (mx - mn)

    def _content_score(self, query):
        qv = self.tfidf.transform([normalize_value(query)])
        return cosine_similarity(self.X_tfidf, qv).flatten()

    def _latent_score(self, query):
        qv = self.tfidf.transform([normalize_value(query)])
        q_emb = self.svd.transform(qv)
        return cosine_similarity(self.prod_embeddings, q_emb).flatten()

    def _weak_ties_boost(self, query, boost_factor=1.0):
        q_terms = [t for t in normalize_value(query).split() if t]
        boost_vec = np.zeros(self.X_tfidf.shape[0], dtype=float)
        for term in q_terms:
            if term in self.tfidf.vocabulary_:
                term_id = self.tfidf.vocabulary_[term]
                term_idf = self.tfidf.idf_[term_id]
                matched_attrs = [a for a in self.attribute_nodes if term in self.G.nodes[a]["attr"]]
                for a in matched_attrs:
                    for p in self.G.neighbors(a):
                        if p.startswith("p_"):
                            pid = p[2:]
                            idx = self.id_to_idx[pid]
                            df_attr = self.attr_df_count.get(a, 1)
                            rarity = math.log((1 + self.N) / (1 + df_attr)) + 1e-9
                            boost_vec[idx] += boost_factor * term_idf * rarity
        return boost_vec

    def _graph_score(self, query, alpha=0.85):
        cont_scores = self._content_score(query)
        top_k = max(1, min(3, len(cont_scores)))
        top_idx = np.argsort(-cont_scores)[:top_k]
        personalization = {}
        for idx in top_idx:
            pid = self.idx_to_id[idx]
            personalization[f"p_{pid}"] = float(cont_scores[idx] + 1e-6)
        q_terms = [t for t in normalize_value(query).split() if t]
        for term in q_terms:
            if term in self.tfidf.vocabulary_:
                matched_attrs = [a for a in self.attribute_nodes if term in self.G.nodes[a]["attr"]]
                for a in matched_attrs:
                    personalization[a] = personalization.get(a, 0.0) + 1.0 * self.tfidf.idf_[self.tfidf.vocabulary_[term]]
        if personalization:
            s = sum(personalization.values())
            personalization = {k: v/s for k, v in personalization.items()}
        else:
            personalization = {p: 1.0/len(self.product_nodes) for p in self.product_nodes}
        pr = nx.pagerank(self.G, alpha=alpha, personalization=personalization, weight="weight", max_iter=200)
        pr_scores = np.zeros(self.X_tfidf.shape[0], dtype=float)
        for p_node in self.product_nodes:
            pid = p_node[2:]
            idx = self.id_to_idx[pid]
            pr_scores[idx] = pr.get(p_node, 0.0)
        return pr_scores

    def recommend(self, query, top_n=5, w_content=0.5, w_latent=0.2, w_graph=0.3):
        if not query.strip():
            return []

        content_scores = self._normalize(self._content_score(query))
        latent_scores = self._normalize(self._latent_score(query))
        graph_scores = self._normalize(self._graph_score(query))
        weak_boost = self._normalize(self._weak_ties_boost(query))

        final = (w_content * content_scores +
                 w_latent * latent_scores +
                 w_graph * graph_scores +
                 0.25 * weak_boost)

        final = self._normalize(final)

        top_idx = np.argsort(-final)[:top_n]
        results = []
        for idx in top_idx:
            pid = self.idx_to_id[idx]
            rec = self.df.loc[pid].to_dict()
            title = rec.get("Model Number") or rec.get("Type") or rec.get("title") or rec.get("text")[:50]
            results.append({
                "id": pid,
                "title": title
            })
        return results



# Main script

def main():
    # 1. Load JSON
    json_path = "/Users/ayushbhandari/Desktop/KnitSpace/data-set.json"   
    with open(json_path, "r") as f:
        records = json.load(f)

    # 2. Fit the engine
    engine = WeakTiesRecommender()
    engine.fit(records)

    # 3. Interactive search
    print("\n🔎 Weak Ties Search Engine Ready!")
    print("Type a query (or 'exit' to quit):\n")

    while True:
        query = input("Query: ")
        if query.lower().strip() in ["exit", "quit"]:
            break
        results = engine.recommend(query, top_n=5)
        print(f"\nResults for: '{query}'\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
        print("\n")

if __name__ == "__main__":
    main()
