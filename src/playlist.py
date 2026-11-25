"""Feature-based playlist sequencing (the dataset has no playlist/transition data, so no Markov model)."""
import numpy as np

ARCS = {"steady": None, "build_up": 0.9, "wind_down": 0.25}


def build_playlist(rec, seed, length=10, arc="steady", pool=300):
    """Greedy path: start at the seed; each step picks, from the seed's nearest neighbours, the unused track that minimises
    cost = distance to previous track (smooth transitions) + 2*|energy - target energy| (follows the energy arc)."""
    df, X = rec.df, rec.X
    cand = list(rec.similar(seed, pool, pool=pool + 1).index)
    end = ARCS[arc] if ARCS[arc] is not None else df.energy[seed]
    path = [seed]
    for k in range(1, length):
        target = df.energy[seed] + (end - df.energy[seed]) * k / (length - 1)
        prev = X[path[-1]]
        cost = [np.linalg.norm(X[c] - prev) / np.sqrt(len(prev)) + 2 * abs(df.energy[c] - target) for c in cand]
        best = cand.pop(int(np.argmin(cost))); path.append(best)
    return df.loc[path]
