# Face Diary – Face Features + Mood Logging

Small project where I log my face over time and try to learn something from it using a pre‑trained computer vision model and some simple ML. The app runs in Streamlit and lets you capture a face entry with your webcam or use a sample image, tag it with a mood (tired / neutral / happy / stressed / other) and a short note, extract numeric face features using Mediapipe FaceMesh, see all entries in a diary table and plot features over time, and run both KMeans clustering on the face feature vectors (unsupervised) and a small RandomForest classifier to predict mood from those features (supervised).

## Tech

- Python  
- Streamlit  
- Mediapipe (FaceMesh)  
- OpenCV (headless)  
- NumPy / pandas  
- scikit‑learn  

## Setup

Install the dependencies (in a virtualenv if you want):

```bash
pip install -r requirements.txt
