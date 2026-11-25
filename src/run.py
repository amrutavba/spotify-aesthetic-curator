"""python -m src.run [--demo] : clean, label moods, cluster, evaluate recommender."""
import argparse, json
import numpy as np
from . import data, moods
from .config import *
from .recommender import Recommender


def main(demo=False):
    df = data.synthetic() if demo else data.load()
    df["mood"] = moods.label_moods(df); df["cluster"], prof = moods.cluster(df)
    PROCESSED.mkdir(exist_ok=True, parents=True); REPORTS.mkdir(exist_ok=True)
    df.to_csv(PROCESSED / "tracks_clean.csv", index=False); prof.to_csv(REPORTS / "cluster_profiles.csv")
    print(f"tracks={len(df)}\n{df.mood.value_counts().to_string()}\n{prof.to_string()}")
    rec = Recommender(df); rng = np.random.default_rng(SEED)
    seeds = rng.choice(len(df), 200, replace=False)
    # Sanity metric (genre is NOT used in similarity): share of top-10 recs having the seed's genre vs random baseline
    hit = np.mean([(rec.similar(s, 10).track_genre == df.track_genre[s]).mean() for s in seeds]) if "track_genre" in df else float("nan")
    base = float((df.track_genre.value_counts(normalize=True) ** 2).sum()) if "track_genre" in df else float("nan")
    print(f"genre hit-rate@10 = {hit:.2f} vs random baseline {base:.2f}")
    (REPORTS / "metrics.json").write_text(json.dumps({"genre_hit_rate_at_10": hit, "random_baseline": base, "n_tracks": len(df)}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true"); main(ap.parse_args().demo)
