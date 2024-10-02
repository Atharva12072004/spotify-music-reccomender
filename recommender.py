import streamlit as st
import pandas as pd
import pickle

# Load the saved recommendation model
with open('recommendation.pkl', 'rb') as f:
    knn = pickle.load(f)  # Load the model directly

# Load the dataset and feature scaling separately
df_cleaned = pd.read_csv(r'C:\Users\ATHARVA\OneDrive\Desktop\gana\spotify dataset.csv')  # Load the dataset
audio_features = ['danceability', 'energy', 'key', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
                  
# Assuming features were scaled during the model training
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cleaned[audio_features])
import random
import streamlit as st

# Streamlit app interface
st.title("Spotify Song Recommendation System")

# Dropdown for Genre
genre_list = df_cleaned['playlist_genre'].dropna().unique()
selected_genre = st.selectbox("Select a Genre", sorted(genre_list))

# Filter playlists based on selected genre
filtered_df_genre = df_cleaned[df_cleaned['playlist_genre'] == selected_genre]

# Dropdown for Playlist Name
playlist_names = filtered_df_genre['playlist_name'].dropna().unique()
selected_playlist = st.selectbox("Select a Playlist Name", sorted(playlist_names))

# Filter subgenres based on selected playlist
filtered_df_playlist = filtered_df_genre[filtered_df_genre['playlist_name'] == selected_playlist]
subgenre_list = filtered_df_playlist['playlist_subgenre'].dropna().unique()
selected_subgenre = st.selectbox("Select a Subgenre", sorted(subgenre_list))

# Number of recommendations
num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

# Get and display recommendations when the button is clicked
if st.button("Recommend Songs"):
    # Filter the dataset based on selected genre, playlist, and subgenre
    user_filtered_df = filtered_df_playlist[filtered_df_playlist['playlist_subgenre'] == selected_subgenre]

    if not user_filtered_df.empty:
        # Reset index to align with scaled_features
        user_filtered_df = user_filtered_df.reset_index()
        # Get the indices of the filtered songs
        song_indices = user_filtered_df.index.tolist()
        # Choose a random song index from the filtered list
        base_song_index = random.choice(song_indices)
        # Get recommendations using kneighbors method
        distances, indices = knn.kneighbors([scaled_features[base_song_index]], n_neighbors=num_recommendations + 1)
        # Exclude the base song from recommendations
        recommended_indices = indices[0][1:]
        recommended_songs = df_cleaned.iloc[recommended_indices]
        st.write(f"Recommendations based on your selection:")
        st.dataframe(recommended_songs[['track_name', 'track_artist', 'playlist_genre']])
    else:
        st.write("No songs found for the selected combination.")
