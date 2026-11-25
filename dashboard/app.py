"""streamlit run dashboard/app.py  (run `python -m src.run` first)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src import playlist
from src.config import *
from src.recommender import Recommender

st.set_page_config("Spotify Aesthetic Curator", layout="wide")


@st.cache_resource
def load():
    df = pd.read_csv(PROCESSED / "tracks_clean.csv")
    return df, Recommender(df)


try:
    df, rec = load()
except FileNotFoundError:
    st.error("Run `python -m src.run` (or `--demo`) first."); st.stop()

st.title("Spotify Aesthetic Curator")
q = st.sidebar.text_input("Search track or artist", "")
hits = rec.search(q) if q else df.sort_values("popularity", ascending=False).head(25)
label = lambda i: f"{df.track_name[i]} - {df.artists[i]}"
seed = st.sidebar.selectbox("Select seed track", list(hits.index), format_func=label)
s = df.loc[seed]
st.sidebar.write(f"**Mood:** {s.mood}  \n**Genre:** {s.get('track_genre', '-')}")
t1, t2, t3, t4 = st.tabs(["Similar songs", "Playlist generator", "Track profile", "Explore"])

with t1:
    c = st.columns(5)
    e = c[0].slider("Energy", 0.0, 1.0, (0.0, 1.0)); v = c[1].slider("Valence", 0.0, 1.0, (0.0, 1.0)); d = c[2].slider("Danceability", 0.0, 1.0, (0.0, 1.0))
    bpm = c[3].slider("Tempo (BPM)", 40, 220, (40, 220)); moods = c[4].multiselect("Mood", sorted(df.mood.unique()))
    mask = df.energy.between(*e) & df.valence.between(*v) & df.danceability.between(*d) & df.tempo.between(*bpm)
    if moods: mask &= df.mood.isin(moods)
    n = st.slider("How many", 5, 30, 10)
    r = rec.similar(seed, n, mask=mask.values, pool=2000)
    st.dataframe(r[["track_name", "artists", "mood", "similarity", "popularity"] + (["track_genre"] if "track_genre" in r else [])].style.format({"similarity": "{:.1%}"}), use_container_width=True)
    if len(r):
        top = r.index[0]; ex = rec.explain(seed, top)
        st.plotly_chart(px.bar(ex, orientation="h", title=f"Why '{df.track_name[top]}' is similar: share of cosine score by feature"), use_container_width=True)

with t2:
    c = st.columns(2); length = c[0].slider("Playlist length", 5, 25, 10); arc = c[1].selectbox("Energy arc", list(playlist.ARCS))
    p = playlist.build_playlist(rec, seed, length, arc)
    st.dataframe(p[["track_name", "artists", "mood", "energy", "valence", "tempo"]].round(2), use_container_width=True)
    st.plotly_chart(px.line(p.reset_index(drop=True), y=["energy", "valence", "danceability"], markers=True, title="Playlist flow"), use_container_width=True)
    st.caption("No playlist transition data in this dataset, so sequencing is feature-based (smooth feature distance + energy arc).")

with t3:
    feats = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"]
    fig = go.Figure(go.Scatterpolar(r=s[feats].astype(float), theta=feats, fill="toself", name=label(seed)))
    for i in rec.similar(seed, 3).index:
        fig.add_trace(go.Scatterpolar(r=df.loc[i, feats].astype(float), theta=feats, name=label(i)))
    st.plotly_chart(fig, use_container_width=True)

with t4:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(df.mood.value_counts().reset_index(), names="mood", values="count", title="Mood distribution", hole=.4), use_container_width=True)
    art = df.groupby("artists").agg(tracks=("popularity", "size"), pop=("popularity", "mean")).query("tracks >= 3").nlargest(15, "pop")
    c2.plotly_chart(px.bar(art["pop"], orientation="h", title="Most popular artists (>=3 tracks)"), use_container_width=True)
    if "track_genre" in df:
        st.plotly_chart(px.bar(df.track_genre.value_counts().head(20), title="Top genres by track count"), use_container_width=True)
    x, y = st.columns(2); fx = x.selectbox("X", AUDIO, 1); fy = y.selectbox("Y", AUDIO, 0)
    st.plotly_chart(px.scatter(df.sample(min(4000, len(df)), random_state=SEED), x=fx, y=fy, color="mood", opacity=.5), use_container_width=True)
    st.plotly_chart(px.imshow(df[AUDIO].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Audio feature correlations"), use_container_width=True)
