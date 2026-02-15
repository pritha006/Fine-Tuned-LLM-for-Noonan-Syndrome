from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import numpy as np

def cosine_confidence(query_embedding, doc_embeddings):
    sims = cosine_similarity([query_embedding], doc_embeddings)
    return float(np.max(sims))

def f1_evaluation(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")
