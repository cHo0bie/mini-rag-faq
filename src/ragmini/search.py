import numpy as np

def cosine_top_k(query_vec, mat, k=5):
    # матрица разреженная; для стабильности нормируем на уровне toarray()
    numer = (mat @ query_vec.T).toarray().ravel()
    A = mat.toarray()
    denom = (np.linalg.norm(A, axis=1) * np.linalg.norm(query_vec.toarray()))
    denom[denom==0] = 1e-9
    scores = numer / denom
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]

def search(q: str, vect, mat, corpus, meta, k=5):
    qv = vect.transform([q])
    idx, scr = cosine_top_k(qv, mat, k=k)
    out = []
    for i, s in zip(idx, scr):
        out.append({**meta[i], "score": float(s), "passage": corpus[i]})
    return out
