import streamlit as st
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

st.title("MediFind")
st.write("Search lung cancer abstracts. Research tool, not medical advice.")

@st.cache_resource
def load():
    df = pd.read_csv("lung_abstracts.csv")
    vecs = np.load("doc_vecs.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    bm25 = BM25Okapi([a.split() for a in df["abstract"]])
    return df, vecs, model, bm25

df, vecs, model, bm25 = load()
query = st.text_input("Enter your search")

if query:
    b = bm25.get_scores(query.lower().split()).argsort()[::-1][:5]
    s = util.cos_sim(model.encode(query), vecs)[0].argsort(descending=True)[:5]
    col1, col2 = st.columns(2)
    col1.subheader("BM25")
    for i in b:
        col1.write(df["title"][int(i)])
    col2.subheader("Semantic")
    for i in s:
        col2.write(df["title"][int(i)])
