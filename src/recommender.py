"""Content-based recommender: standardise -> weight -> cosine nearest neighbours."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .config import *


class Recommender:
    """cosine_similarity(a, b) = a.b / (|a||b|) on standardised, weighted audio features (angle between taste vectors)."""
    def __init__(self, df, weights=WEIGHTS):
        self.df = df.reset_index(drop=True)
        self.w = np.array([weights[c] for c in AUDIO])
        self.X = StandardScaler().fit_transform(self.df[AUDIO]) * self.w
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(self.X)

    def search(self, text, limit=25):
        m = self.df.track_name.str.contains(text, case=False, regex=False) | self.df.artists.str.contains(text, case=False, regex=False)
        return self.df[m].sort_values("popularity", ascending=False).head(limit)

    def similar(self, idx, n=10, mask=None, pool=500):
        """Top-n similar tracks; optional boolean `mask` (filters) applied to candidates; excludes the seed and same-title duplicates."""
        d, i = self.nn.kneighbors(self.X[[idx]], n_neighbors=min(pool, len(self.df)))
        out = pd.DataFrame({"idx": i[0], "similarity": 1 - d[0]})
        out = out[out.idx != idx]
        if mask is not None:
            out = out[np.asarray(mask)[out.idx.values]]
        out = out.head(n)
        return self.df.loc[out.idx].assign(similarity=out.similarity.values)

    def explain(self, i, j):
        """Share of the cosine score contributed by each feature (higher = features that make the two tracks alike)."""
        c = self.X[i] * self.X[j] / (np.linalg.norm(self.X[i]) * np.linalg.norm(self.X[j]))
        return pd.Series(c, index=AUDIO).sort_values(ascending=False)
