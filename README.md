# Spam Detection Web App

A machine learning-based web application that detects whether a message is Spam or Not Spam.

## Features
- Spam / Not Spam prediction
- Block spam messages
- Dataset visualization (spam vs ham)
- History tracking

## Tech Stack
- Python
- Flask
- Scikit-learn
- Pandas

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the app:
python app.py

3. Open in browser:
http://127.0.0.1:5000

## Project Structure
- app.py → Flask backend
- spam_project.py → Model training
- templates/ → HTML UI
- spam.csv → Dataset

## Future Improvements
- Add spam probability score
- Search/filter dataset
- Deploy online
