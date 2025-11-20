Music Genre Classification & Song Recommendation System

A machine-learning powered system that predicts the genre of an uploaded audio file and generates personalized song recommendations using GPT-Neo and the Spotify Web API.

Project Structure
├── app.py                   # Streamlit app (UI + Prediction flow)
├── genre_classifier.pkl     # Pre-trained ML model
├── scaler.pkl               # StandardScaler object
├── requirements.txt         # Python dependencies
└── main_musicgenre.yml      # GitHub Actions workflow for Azure deployment

Core Features
Feature	Description
Audio Upload	Supports MP3, WAV, OPUS file formats
Feature Extraction	Extracts MFCC + Chroma features using Librosa
Genre Prediction	Predicts genre using a trained ML model
AI Song Recommendations	GPT-Neo generates 5 matching song suggestions
Spotify Integration	Spotipy fetches real Spotify track links
Weaviate Support	Optional vector database for metadata storage
System Workflow

User uploads an audio file

Librosa extracts MFCC + Chroma audio features

Features are scaled with StandardScaler

Genre is predicted using the ML classifier

GPT-Neo recommends 5 suitable songs

Spotify API fetches track URLs

Streamlit UI displays:

Predicted genre

Extracted features

Recommended songs with links

How to Run the Project
pip install -r requirements.txt
streamlit run app.py


Upload any audio file and instantly view predictions + recommendations.

Required Environment Variables

Set these before running the project:

SPOTIPY_CLIENT_ID

SPOTIPY_CLIENT_SECRET

WEAVIATE_URL

WEAVIATE_API_KEY

Deployment

This project includes Azure Web App deployment using GitHub Actions.

main_musicgenre.yml handles:

Installing dependencies

Packaging the project

Azure authentication

Automatic deployment on push to main branch

About the Developer

Author: Parikshith DS
Email: parikshithds.1si21ad037@gmail.com
GitHub: https://github.com/parikshith49
