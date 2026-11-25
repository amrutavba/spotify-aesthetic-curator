"""Spotify Tracks Dataset (~114k tracks, 125 genres; Kaggle/HuggingFace 'maharshipandya/spotify-tracks-dataset')."""
import urllib.request
import numpy as np
import pandas as pd
from .config import *


def load():
    """Use data/raw/tracks.csv; download it if missing (or place the Kaggle CSV there manually)."""
    path = RAW / "tracks.csv"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(URL, path)
        except Exception as e:
            raise RuntimeError(f"Download failed ({e}). Get 'Spotify Tracks Dataset' from Kaggle "
                               f"(maharshipandya) and save as {path}.") from e
    return clean(pd.read_csv(path))


def clean(df):
    """Drop index column, missing values, impossible values and duplicate tracks (keep most popular copy)."""
    df = df.drop(columns=[c for c in df if c.startswith("Unnamed")]).dropna(subset=AUDIO + ["track_name", "artists"])
    df = df[(df.tempo > 0) & df[["danceability", "energy", "valence", "acousticness"]].apply(lambda s: s.between(0, 1)).all(axis=1)]
    df = df.sort_values("popularity", ascending=False).drop_duplicates("track_id").drop_duplicates(["track_name", "artists"])
    return df.reset_index(drop=True)


def synthetic(n=6000, seed=SEED):
    """Genre-structured synthetic tracks (DEMO/TESTS ONLY)."""
    rng = np.random.default_rng(seed)
    genres = {"edm": (.75, .85, .5), "acoustic": (.45, .25, .5), "hip-hop": (.8, .65, .55), "metal": (.4, .95, .3), "chill": (.55, .35, .6), "pop": (.65, .7, .65)}
    g = rng.choice(list(genres), n); m = np.array([genres[x] for x in g])
    c = lambda x: np.clip(x, 0, 1)
    df = pd.DataFrame({"track_id": [f"id{i}" for i in range(n)], "track_name": [f"Track {i}" for i in range(n)],
                       "artists": [f"Artist {i % 400}" for i in range(n)], "track_genre": g, "popularity": rng.integers(0, 100, n),
                       "danceability": c(rng.normal(m[:, 0], .1)), "energy": c(rng.normal(m[:, 1], .1)), "valence": c(rng.normal(m[:, 2], .15)),
                       "acousticness": c(rng.normal(1 - m[:, 1], .15)), "instrumentalness": c(rng.beta(.3, 3, n)), "speechiness": c(rng.beta(1, 12, n)),
                       "liveness": c(rng.beta(1.5, 8, n)), "loudness": -20 + 16 * m[:, 1] + rng.normal(0, 2, n), "tempo": rng.normal(60 + 90 * m[:, 1], 12).clip(50, 200)})
    return clean(df)
