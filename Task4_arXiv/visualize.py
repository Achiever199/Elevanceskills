"""
visualize.py
------------
Concept-visualization charts for the "Explore the Field" tab:
  - category distribution (which cs.* subfields dominate the loaded corpus)
  - papers-per-year trend
  - a 2D concept map: TF-IDF vectors of all papers projected to 2D via PCA
    and colored by primary category, so semantically related papers cluster
    visually - a lightweight, fully local stand-in for an embedding atlas.
"""

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def category_distribution_chart(df: pd.DataFrame):
    counts = {}
    for cats in df["categories"].dropna():
        for tag in str(cats).split():
            if tag.startswith("cs."):
                counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return None
    chart_df = pd.DataFrame(sorted(counts.items(), key=lambda x: -x[1]), columns=["category", "count"])
    fig = px.bar(chart_df, x="category", y="count", title="Papers per CS Subfield")
    fig.update_layout(xaxis_tickangle=-40, height=400)
    return fig


def papers_per_year_chart(df: pd.DataFrame):
    if "year" not in df.columns or df["year"].dropna().empty:
        return None
    counts = df["year"].dropna().astype(int).value_counts().sort_index()
    chart_df = pd.DataFrame({"year": counts.index, "count": counts.values})
    fig = px.line(chart_df, x="year", y="count", markers=True, title="Publication Volume Over Time")
    fig.update_layout(height=350)
    return fig


def concept_map(df: pd.DataFrame, max_points: int = 300):
    """2D scatter of papers positioned by textual similarity (TF-IDF + PCA),
    colored by primary category. Hovering shows the title."""
    plot_df = df.head(max_points).copy()
    docs = (plot_df["title"].fillna("") + ". " + plot_df["abstract"].fillna("")).tolist()
    if len(docs) < 3:
        return None

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(docs).toarray()

    n_components = min(2, matrix.shape[0] - 1, matrix.shape[1])
    if n_components < 2:
        return None

    coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="primary_category",
        hover_data={"title": True, "x": False, "y": False, "primary_category": True},
        title="Concept Map: Papers Clustered by Textual Similarity",
    )
    fig.update_layout(height=500)
    return fig
