
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Booktwin", page_icon="📖", layout="centered")

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
h1 { font-family: "DM Serif Display", serif !important; font-size: 48px !important; font-weight: 400 !important; color: #1a1a1a !important; }
.subtitle { font-size: 17px; color: #666; margin-bottom: 40px; font-weight: 300; }
footer {visibility: hidden;} #MainMenu {visibility: hidden;} header {visibility: hidden;}
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv('booktok_final.csv')
    embeddings = np.load('booktok_embeddings.npy')
    return df, embeddings

model = load_model()
df, embeddings = load_data()

st.markdown("<h1>Booktwin</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Type a book you loved. We will find your next one.</p>', unsafe_allow_html=True)

query = st.text_input("", placeholder="e.g. A Court of Thorns and Roses...", label_visibility="collapsed")

if st.button("Find my booktwin") and query:
    matches = df[df["title"].str.lower().str.contains(query.lower())]
    
    if len(matches) == 0:
        st.markdown("### We dont have that book yet")
        st.markdown("Try another title!")
    else:
        idx = matches.index[0]
        found_title = df.iloc[idx]["title"]
        similarities = cosine_similarity([embeddings[idx]], embeddings)[0]
        similar_indices = similarities.argsort()[::-1][1:6]
        
        st.markdown(f"**Because you loved {found_title}:**")
        st.markdown("---")
        
        for sim_idx in similar_indices:
            book = df.iloc[sim_idx]
            score = similarities[sim_idx]
            thumbnail = book.get("thumbnail", "")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if thumbnail and str(thumbnail) != "nan":
                    st.image(thumbnail, width=80)
                else:
                    st.markdown("📖")
            with col2:
                st.markdown(f"**{book['title']}**")
                st.markdown(f"*{book['authors']}*")
                st.markdown(f"{str(book['description'])[:140]}...")
                st.markdown(f"<span style='color:#a07850;font-size:12px'>{int(score*100)}% match</span>", unsafe_allow_html=True)
            st.markdown("---")
