# Spotify Aesthetic Curator

Content-based music recommender, mood/aesthetic classifier and playlist generator on Spotify audio features, with a Streamlit app.

## Problem & objective
Given a track, recommend similar songs from audio characteristics, label tracks with understandable moods, and sequence them into a playlist with a controllable energy arc.

## Dataset
**Spotify Tracks Dataset** (~114k tracks, 125 genres, 9+ audio features; Kaggle `maharshipandya/spotify-tracks-dataset`, also hosted on Hugging Face; CC-BY/ODbL-style terms — check the dataset page). The Spotify Million Playlist Dataset needs an application and licence, and the Spotify API no longer exposes audio features to new apps, so a static public CSV is used. `python -m src.run` tries to download `data/raw/tracks.csv`; otherwise download it manually. `--demo` uses synthetic data for offline testing only.

## Methodology
1. **Cleaning:** drop missing/invalid rows, de-duplicate by `track_id` and (title, artists).
2. **Recommender:** standardise 9 audio features → multiply by interpretable weights (mood features ×1.5, liveness ×0.4) → **cosine similarity** via brute-force `NearestNeighbors`. `explain()` shows each feature's share of the cosine score.
3. **Moods (rule-based, first match wins):** Dance-oriented (dance ≥ .7 & energy ≥ .6), Energetic (energy ≥ .7), Upbeat (valence ≥ .6 & energy ≥ .5), Melancholic (valence < .35 & energy < .6), Acoustic (acousticness ≥ .6), Relaxed (energy < .45), else Balanced. **K-Means (k=6)** gives a data-driven alternative with profile table.
4. **Playlist sequencing:** greedy path through the seed's 300 nearest neighbours minimising distance to the previous track plus a penalty for deviating from a target energy arc (steady / build-up / wind-down). No playlist data exists in this dataset, so a transition (Markov) model is not used.
5. **Evaluation:** genre is *not* used in similarity, so **genre hit-rate@10** (share of recommendations in the seed's genre) vs the random baseline is a sanity check, not a measure of taste.

## Structure
`src/{config,data,moods,recommender,playlist,run}.py` · `dashboard/app.py` · `tests/test_music.py`

## Run
```bash
pip install -r requirements.txt
python -m src.run          # add --demo offline
pytest -q
streamlit run dashboard/app.py
```

## Results (from `reports/metrics.json`)
| Metric | Value |
|---|---|
| Tracks after cleaning | |
| Genre hit-rate@10 | |
| Random baseline | |

## Limitations
Audio-feature similarity ignores lyrics, language, era and listening behaviour (no collaborative signal); rule thresholds are judgement-based; genre hit-rate is a proxy.

## Future improvements
Hybrid with collaborative filtering (needs listening data), A/B-style user feedback loop, approximate NN (FAISS/Annoy) for scale.

**Repo name:** `spotify-aesthetic-curator` · **Topics:** `recommendation-system` `cosine-similarity` `content-based-filtering` `kmeans` `music` `scikit-learn` `streamlit`
