import numpy as np, pytest
from src import data, moods, playlist
from src.config import *
from src.recommender import Recommender


@pytest.fixture(scope="module")
def rec():
    df = data.synthetic(1500); df["mood"] = moods.label_moods(df); return Recommender(df)


def test_clean_no_duplicates_or_invalid():
    df = data.synthetic(500)
    assert not df.duplicated(["track_name", "artists"]).any() and df.tempo.gt(0).all() and df[AUDIO].notna().all().all()


def test_recommendations_exclude_seed_and_sorted(rec):
    r = rec.similar(0, 10)
    assert 0 not in r.index and len(r) == 10 and r.similarity.is_monotonic_decreasing and r.similarity.max() <= 1 + 1e-9


def test_self_similarity_is_one(rec):
    assert rec.explain(3, 3).sum() == pytest.approx(1.0)


def test_filter_mask_respected(rec):
    mask = (rec.df.energy > 0.7).values
    assert (rec.similar(0, 10, mask=mask).energy > 0.7).all()


def test_playlist_length_unique_and_arc(rec):
    seed = int(rec.df.energy.idxmin())                       # low-energy seed so a build-up arc must rise
    p = playlist.build_playlist(rec, seed, 8, "build_up")
    assert len(p) == 8 and p.index.is_unique and p.energy.iloc[-3:].mean() > p.energy.iloc[:3].mean()


def test_moods_cover_all_tracks(rec):
    assert rec.df.mood.notna().all() and rec.df.mood.nunique() >= 3
