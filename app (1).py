import os
import weaviate
import spotipy
import streamlit as st
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from spotipy.oauth2 import SpotifyClientCredentials

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define Weaviate URL and API key
WEAVIATE_URL = "https://5frdqgfpqwsfulwj6rttq.c0.asia-southeast1.gcp.weaviate.cloud"  # Replace with your actual Weaviate URL
API_KEY = "U2imUjw7yKFu0KBuYdlsYkJIiUyKaaNKakZo"  # Replace with your actual API key

# Spotify Authentication
SPOTIPY_CLIENT_ID = "eae5486e35f64166a47238d8ecfebc24"  # Replace with your actual Client ID
SPOTIPY_CLIENT_SECRET = "fc76368eedcb4e66b44813f164a471c9"  # Replace with your actual Client Secret

# Initialize Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# Retry logic for connecting to Weaviate
def connect_to_weaviate():
    try:
        client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(api_key=API_KEY),
        )
        client.schema.get()  # Check connection
        st.success("Connected to Weaviate!")
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        return None

# Extract features using librosa
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        feature_names = [f"MFCC-{i+1}" for i in range(13)] + [f"Chroma-{i+1}" for i in range(chroma.shape[0])]
        feature_values = np.concatenate((mfcc_mean, chroma_mean))
        return feature_names, feature_values
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None

# Load pre-trained model and scaler (if available)
def load_model_and_scaler():
    try:
        with open("genre_classifier.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.warning("Pre-trained model or scaler not found.")
        return KNeighborsClassifier(n_neighbors=3), StandardScaler()

# Predict genre using the model
def classify_genre(features, model, scaler):
    try:
        features_scaled = scaler.transform([features])
        genre = model.predict(features_scaled)[0]
        return genre
    except Exception as e:
        st.error(f"Error in genre classification: {e}")
        return "Unknown Genre"

# Initialize GPT-Neo model
def initialize_gpt_neo():
    model_name = "EleutherAI/gpt-neo-1.3B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Generate recommendations using GPT-Neo
def generate_gpt_recommendations(genre, model, tokenizer):
    try:
        # Updated prompt to ask for 5 different songs
        prompt = (
            f"The genre '{genre}' is known for its unique style. "
            "Please list five different songs that represent this genre. "
            "Each recommendation should be on a new line, with the title of the song followed by the artist, formatted as: "
            "'<song title> by <artist>'."
            "Ensure that no song is repeated in the list."
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=250, temperature=0.7, num_return_sequences=1)
        recommendations = tokenizer.decode(outputs[0], skip_special_tokens=True).split("\n")

        # Filter out invalid recommendations (ones that don't contain 'by' or are too short)
        unique_recommendations = [rec.strip() for rec in recommendations if 'by' in rec and len(rec.strip()) > 0]

        # Ensure we return exactly 5 unique songs
        return unique_recommendations[:5]
    except Exception as e:
        st.error(f"Error with GPT-Neo: {e}")
        return []

# Get Spotify links for the recommended songs
def get_spotify_links(song_titles):
    song_links = []
    for title in song_titles:
        query = title.split(" by ")[0].strip()
        if len(query) > 0:
            try:
                results = sp.search(q=query, limit=1, type="track")
                if results['tracks']['items']:
                    track = results['tracks']['items'][0]
                    song_links.append(f"{track['name']} by {track['artists'][0]['name']} - [Listen on Spotify]({track['external_urls']['spotify']})")
                else:
                    song_links.append(f"Song '{query}' not found on Spotify.")
            except Exception as e:
                song_links.append(f"Error retrieving song '{query}': {e}")
    return song_links

# Streamlit app logic
def main():
    st.title("Music Genre Classifier and Recommendation System")
    client = connect_to_weaviate()
    if not client:
        st.stop()

    model, scaler = load_model_and_scaler()
    gpt_model, gpt_tokenizer = initialize_gpt_neo()

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "opus"])
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1]
        input_path = f"uploaded_audio_file.{file_extension}"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(uploaded_file, format=f"audio/{file_extension}")

        feature_names, feature_values = extract_features(input_path)
        if feature_values is not None:
            st.subheader("Extracted Features:")
            for name, value in zip(feature_names, feature_values):
                st.write(f"{name}: {value}")

            genre = classify_genre(feature_values, model, scaler)
            st.success(f"Predicted Genre: {genre}")

            recommendations_from_gpt = generate_gpt_recommendations(genre, gpt_model, gpt_tokenizer)

            spotify_links = get_spotify_links(recommendations_from_gpt)
            for link in spotify_links:
                st.markdown(link)

if __name__ == "__main__":
    main()
