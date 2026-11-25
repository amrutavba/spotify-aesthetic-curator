from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, REPORTS = ROOT / "data/raw", ROOT / "data/processed", ROOT / "reports"
SEED, N_CLUSTERS = 42, 6
URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
# Feature weights for similarity (1 = neutral). Mood-defining features count more.
WEIGHTS = {"danceability": 1.5, "energy": 1.5, "valence": 1.5, "acousticness": 1.2, "tempo": 1.0, "loudness": 0.8,
           "speechiness": 0.6, "instrumentalness": 0.6, "liveness": 0.4}
