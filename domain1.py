from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
from collections import Counter

# --------------------------
# NLTK setup
# --------------------------
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")

# --------------------------
# Load embedding model
# --------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Example paragraph
# --------------------------
paragraph = """
Cristiano Ronaldo scored a hat-trick in the Champions League final.  
Lionel Messi won his eighth Ballon d'Or after a stellar season.  
The stock market crashed due to inflation concerns and rising interest rates.  
Apple released its latest iPhone with advanced AI features.  
Google announced breakthroughs in quantum computing research.  

"""

# --------------------------
# Step 1: Tokenization
# --------------------------
sentences = nltk.sent_tokenize(paragraph)
embeddings = model.encode(sentences)

# --------------------------
# Step 2: Clustering
# --------------------------
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
centroids = kmeans.cluster_centers_

# --------------------------
# Step 3: Predefined clean domains
# --------------------------
candidate_domains = [
    "Sports", "Animals", "Health", "Food", "Politics", "Technology", "Education",
    "Travel", "Entertainment", "Economy", "Science", "Environment", "History",
    "Geography", "Culture", "Philosophy", "Religion", "Art", "Literature", "Music",
    "Law", "Military", "Space", "Agriculture", "Business", "Finance", "Media",
    "Communication", "Transportation", "Fashion", "Psychology", "Sociology",
    "Anthropology", "Linguistics", "Mathematics", "Physics", "Chemistry", "Biology",
    "Astronomy", "Engineering", "Architecture", "Medicine", "Pharmacy", "Public Policy",
    "Government", "International Relations", "Security", "Tourism", "Lifestyle", "Family",
    "Gender Studies", "Cultural Studies", "Ethics", "AI and Data Science", "Computing",
    "Robotics", "Energy", "Environment & Climate", "History of Science", "Archaeology",
    "Mythology", "Performing Arts", "Visual Arts"
]

domain_embeddings = model.encode(candidate_domains)

# --------------------------
# Step 4: Assign clusters → domain labels
# --------------------------
meaningful_labels = {}
for i in range(num_clusters):
    sims = cosine_similarity([centroids[i]], domain_embeddings)
    best_index = np.argmax(sims)
    meaningful_labels[i] = candidate_domains[best_index]

print("Cluster → Domain Mapping:", meaningful_labels)

# --------------------------
# Step 5: Sentence → Domain mapping
# --------------------------
print("\nSentence-level Domain Classification:\n")
sentence_domains = []
for i, sentence in enumerate(sentences):
    cluster_id = labels[i]
    domain = meaningful_labels[cluster_id]
    sentence_domains.append(domain)
    print(f"- {sentence}  -->  {domain}")

# --------------------------
# Step 6: Domain percentage distribution
# --------------------------
domain_counts = Counter(sentence_domains)
total_sentences = len(sentences)

print("\nDomain Percentage Distribution:\n")
for domain, count in domain_counts.items():
    percentage = (count / total_sentences) * 100
    print(f"{domain}: {percentage:.2f}%")
