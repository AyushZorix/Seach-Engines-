from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
from collections import Counter

# --------------------------
# Download required NLTK resource
# --------------------------
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --------------------------
# Load a powerful sentence embedding model
# --------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Step 1: Input paragraph
# --------------------------
paragraph = """
cat is moving around . i dont know why .
"""

# --------------------------
# Step 2: Split into sentences and get embeddings
# --------------------------
sentences = nltk.sent_tokenize(paragraph)
embeddings = model.encode(sentences)

# --------------------------
# Step 3: Cluster sentences into domains
# --------------------------
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
centroids = kmeans.cluster_centers_

# --------------------------
# Step 4:  Assign Meaningful Labels to Clusters
# --------------------------
candidate_domains = [
    "Technology & Business",
    "Food & Health",
    "Sports & Entertainment",
    "Politics & History",
    "Animal"
]
domain_embeddings = model.encode(candidate_domains)

meaningful_labels = {}
for i in range(num_clusters):
    similarities = cosine_similarity([centroids[i]], domain_embeddings)
    best_domain_index = np.argmax(similarities)
    meaningful_labels[i] = candidate_domains[best_domain_index]

# --------------------------
# Step 5: User Interaction
# --------------------------
print("Paragraph:\n", paragraph)
print("Discovered Semantic Domains:", list(set(meaningful_labels.values())))

word = input("\nEnter a word to analyze its context: ").strip()

# --------------------------
# Step 6: Calculate and Display Domain Percentages (The New Logic)
# --------------------------
target_indices = [i for i, s in enumerate(sentences) if word.lower() in s.lower()]

if target_indices:
    # Find the domain for each sentence where the word appears
    occurrence_domains = []
    for i in target_indices:
        cluster_id = labels[i]
        domain = meaningful_labels[cluster_id]
        occurrence_domains.append(domain)

    # Count how many times the word appeared in each domain
    domain_counts = Counter(occurrence_domains)
    total_occurrences = len(target_indices)

    print(f"\nDomain distribution for the word '{word}':\n")
    # Calculate and print the percentage for each domain
    for domain, count in domain_counts.items():
        percentage = (count / total_occurrences) * 100
        print(f"-> {domain}: {percentage:.2f}%")

else:
    print(f"Word '{word}' not found in the paragraph.")