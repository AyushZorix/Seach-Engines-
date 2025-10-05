import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Step 0: Domain classifier
# -------------------------------
domain_db = {
    "sports": ["cricket", "football", "IPL", "World Cup"],
    "bollywood": ["movie", "actor", "bollywood", "film"],
    "tech": ["AI", "ML", "computer", "Python"]
}

def classify_domain(query):
    query = query.lower()
    scores = {domain: sum(query.count(kw.lower()) for kw in keywords) 
              for domain, keywords in domain_db.items()}
    if all(score == 0 for score in scores.values()):
        return "unknown"  # no domain matched
    return max(scores, key=scores.get)

# -------------------------------
# Step 1: Tokenization + embeddings
# -------------------------------
def tokenize(query):
    return query.split()

def embed_tokens(tokens, embedding_dim=8):
    return torch.randn(len(tokens), embedding_dim)  # seq_len x embedding_dim

# -------------------------------
# Step 2: Top-K features (ensure fixed embedding dim)
# -------------------------------
def top_k_features(embeddings, K=8):
    seq_len, embed_dim = embeddings.size()
    if embed_dim < K:
        pad = torch.zeros(seq_len, K - embed_dim)
        embeddings = torch.cat([embeddings, pad], dim=1)
    elif embed_dim > K:
        embeddings = embeddings[:, :K]
    return embeddings

# -------------------------------
# Step 3: Positional encoding
# -------------------------------
def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = torch.tensor(pos / (10000 ** (i / d_model)), dtype=torch.float32)
            PE[pos, i] = torch.sin(angle)
            if i + 1 < d_model:
                PE[pos, i + 1] = torch.cos(angle)
    return PE

# -------------------------------
# Step 4: Encoder
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, embed_dim, nhead=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # seq_len x batch=1 x embed_dim
        attn_output, _ = self.self_attn(x, x, x)
        return attn_output.squeeze(1)  # seq_len x embed_dim

# -------------------------------
# Step 5: Decoder
# -------------------------------
class Decoder(nn.Module):
    def __init__(self, embed_dim, nhead=2, num_classes=3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, tgt, memory):
        tgt = tgt.unsqueeze(1)
        memory = memory.unsqueeze(1)
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt2, _ = self.cross_attn(tgt2, memory, memory)
        out = self.fc(tgt2.mean(dim=0))
        return F.softmax(out, dim=0)

# -------------------------------
# Step 6: Full pipeline
# -------------------------------
def full_pipeline(query, top_K=8):
    print(f"\n=== Input Query ===\n{query}\n")
    
    # Step 0 - Pre-classify domain
    domain_pred = classify_domain(query)
    print(f"Step 0 - Domain Classifier (Before LLM): {domain_pred}")
    
    # Step 1 - Tokenization + embeddings
    tokens = tokenize(query)
    print(f"Step 1 - Tokens: {tokens}")
    emb = embed_tokens(tokens, embedding_dim=top_K)
    
    # Step 2 - Top-K features
    topk_emb = top_k_features(emb, K=top_K)
    print(f"Step 2 - Top-K Features Shape: {topk_emb.shape}")
    
    # Step 3 - Positional encoding
    PE = positional_encoding(topk_emb.size(0), topk_emb.size(1))
    topk_emb = topk_emb + PE
    
    # Step 4 - Encoder
    encoder = Encoder(embed_dim=top_K)
    memory = encoder(topk_emb)
    
    # Step 5 - Decoder
    decoder = Decoder(embed_dim=top_K, num_classes=len(domain_db))
    output_probs = decoder(topk_emb, memory)
    
    predicted_idx = torch.argmax(output_probs).item()
    predicted_domain_from_model = list(domain_db.keys())[predicted_idx]
    print(f"Step 3 - Domain from Model (After LLM): {predicted_domain_from_model}")
    
    # Step 6 - Hallucination check
    if predicted_domain_from_model == domain_pred:
        print("✅ Domain matches! No hallucination.")
    else:
        print("⚠️ Hallucination detected!")
    
    print(f"Step 4 - Model Probabilities: {output_probs.detach().numpy()}")
    return predicted_domain_from_model

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    query = "virat kohli is a human ."
    full_pipeline(query)
