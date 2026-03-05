
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

GOOGLE_BOOKS_API_KEY = st.secrets["GOOGLE_BOOKS_API_KEY"]

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

def assign_genre(desc, cats, title):
    text = (str(cats) + ' ' + str(desc) + ' ' + str(title)).lower()
    if any(x in text for x in ['romance', 'love story', 'falling in love', 'enemies to lovers']):
        if any(x in text for x in ['fantasy', 'fae', 'magic', 'dragon', 'kingdom']):
            return 'romantasy'
        return 'romance'
    if any(x in text for x in ['fantasy', 'magic', 'wizard', 'dragon', 'fae', 'kingdom', 'quest']):
        if any(x in text for x in ['young adult', 'ya', 'teen']):
            return 'ya_fantasy'
        return 'epic_fantasy'
    if any(x in text for x in ['science fiction', 'sci-fi', 'space', 'alien', 'dystopian']):
        return 'scifi'
    if any(x in text for x in ['thriller', 'mystery', 'detective', 'murder', 'crime']):
        return 'thriller_mystery'
    if any(x in text for x in ['horror', 'ghost', 'haunted', 'supernatural']):
        return 'horror'
    if any(x in text for x in ['historical', 'century', 'war', 'medieval']):
        return 'historical_fiction'
    return 'general_fiction'

def fetch_book_live(query):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&key={GOOGLE_BOOKS_API_KEY}&maxResults=1"
    response = requests.get(url)
    data = response.json()
    if 'items' not in data:
        return None
    book = data['items'][0]['volumeInfo']
    return {
        'title': book.get('title', ''),
        'authors': ', '.join(book.get('authors', [])),
        'description': book.get('description', ''),
        'categories': ', '.join(book.get('categories', [])),
        'thumbnail': book.get('imageLinks', {}).get('thumbnail', '')
    }

model = load_model()
df, embeddings = load_data()

st.markdown("<h1>Booktwin</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Type a book you loved. We will find your next one.</p>', unsafe_allow_html=True)

query = st.text_input("Search", placeholder="e.g. A Court of Thorns and Roses...", label_visibility="collapsed")

if st.button("Find my booktwin") and query:
    matches = df[df["title"].str.lower().str.contains(query.lower())]
    
    if len(matches) == 0:
        with st.spinner("Looking up that book..."):
            live_book = fetch_book_live(query)
        
        if live_book is None or not live_book['description']:
            st.markdown("### Could not find that book. Try the full title!")
        else:
            genre = assign_genre(live_book['description'], live_book['categories'], live_book['title'])
            text_blob = genre + ' ' + live_book['title'] + ' by ' + live_book['authors'] + '. ' + live_book['description']
            live_embedding = model.encode([text_blob])
            
            # Filter by same genre first, fall back to all if too few
            genre_mask = df['genre'] == genre
            if genre_mask.sum() >= 5:
                genre_indices = df[genre_mask].index.tolist()
                genre_embeddings = embeddings[genre_indices]
                sims = cosine_similarity(live_embedding, genre_embeddings)[0]
                top_local = sims.argsort()[::-1][:6]
                similar_indices = [genre_indices[i] for i in top_local]
            else:
                sims = cosine_similarity(live_embedding, embeddings)[0]
                similar_indices = sims.argsort()[::-1][:6]
            
            st.markdown(f"**Because you loved {live_book['title']}:**")
            st.markdown("---")
            for sim_idx in similar_indices:
                book = df.iloc[sim_idx]
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
                    st.markdown(f"<span style='color:#a07850;font-size:12px'>{book['genre']}</span>", unsafe_allow_html=True)
                st.markdown("---")
    else:
        idx = matches.index[0]
        found_title = df.iloc[idx]["title"]
        genre = df.iloc[idx].get("genre", "general_fiction")
        
        genre_mask = df['genre'] == genre
        if genre_mask.sum() >= 5:
            genre_indices = [i for i in df[genre_mask].index.tolist() if i != idx]
            genre_embeddings = embeddings[genre_indices]
            sims = cosine_similarity([embeddings[idx]], genre_embeddings)[0]
            top_local = sims.argsort()[::-1][:6]
            similar_indices = [genre_indices[i] for i in top_local]
        else:
            sims = cosine_similarity([embeddings[idx]], embeddings)[0]
            similar_indices = [i for i in sims.argsort()[::-1] if i != idx][:6]
        
        st.markdown(f"**Because you loved {found_title}:**")
        st.markdown("---")
        for sim_idx in similar_indices:
            book = df.iloc[sim_idx]
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
                st.markdown(f"<span style='color:#a07850;font-size:12px'>{book['genre']}</span>", unsafe_allow_html=True)
            st.markdown("---")
