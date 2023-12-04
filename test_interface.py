import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load your perfume data
df = pd.read_csv("perfume_data.csv", encoding="ISO-8859-1")

# Preprocessing --------------------------------------------------------

# Fix column name and combine Brand and Name
df.rename(columns={"ï»¿Name": "Name"}, inplace=True)
df["Name"] = df["Brand"] + " - " + df["Name"]

# Drop irrelevant columns
df.drop(labels=["Description", "Image URL", "Brand"], axis=1, inplace=True)

# Check for missing values and handle them
print(f"Missing values in Notes column: {df.Notes.isnull().sum()}")
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)

# Filter out non-perfume types
words = ["Perfume Oil", "Extrait", "Travel", "Hair", "Body", "Hand", "Intense", "Intensivo", "Oil"]

index_to_drop = []
for index, name in enumerate(df.Name):
    if any(word.lower() in name.lower() for word in words):
        index_to_drop.append(index)

df.drop(index_to_drop, axis=0, inplace=True)
df.reset_index(inplace=True, drop=True)

print(f"Final number of perfumes: {df.shape[0]}")

# ---------------------------------------------------------------------

# Model and Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
note_embeddings = model.encode(df["Notes"].to_list(), show_progress_bar=True)

# Define recommendation function
def recommend_perfumes(perfume, top_n=5):
    perfume_embedding = model.encode([perfume])[0]
    cosine_scores = cosine_similarity(perfume_embedding.reshape(1, -1), note_embeddings)
    top_indices = cosine_scores.argsort()[0][-top_n:]
    recommendations = []
    for i in top_indices:
        recommendations.append({"perfume": df.loc[i, "Name"], "score": cosine_scores[0][i]})
    return recommendations

# Streamlit interface ------------------------------------------------

st.title("Perfume Recommender")

user_perfume = st.text_input("Enter a perfume you like:", placeholder="e.g., Jo Malone - English Pear & Freesia")

if st.button("Recommend perfumes"):
    if user_perfume:
        recommendations = recommend_perfumes(user_perfume)
        st.header("Recommended Perfumes:")
        for i, recommendation in enumerate(recommendations):
            st.write(f"{i+1}. {recommendation['perfume']} (Score: {recommendation['score']:.2f})")
            try:
                image_url = f"https://images.google.com/search?q={recommendation['perfume'].replace(' ','+')}+perfume&tbm=isch"
                st.image(image_url, width=200)
            except:
                pass
    else:
        st.warning("Please enter a perfume to get recommendations!")

# ---------------------------------------------------------------------

