from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
import joblib

app = Flask(__name__)
CORS(app)

# Load necessary resources
with open('api/skills_list.pkl', 'rb') as file:
    skills_list = pickle.load(file)

vectorizer = joblib.load('api/tfidf_vectorizer.pkl')
cosine_similarity_matrix = joblib.load('api/cosine_similarity_matrix.pkl')

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_skills(skills):
    # Tokenize text
    tokens = word_tokenize(skills.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

    # Join tokens back into a string
    return ' '.join(tokens)

# Function to extract text from a PDF file
def extract_text_from_pdf(file_stream):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file_stream)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to extract skills from text
def extract_skills_from_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    
    # Filter tokens that are skills and ensure uniqueness
    skills = {token for token in tokens if token in skills_list}
    return list(skills)

# Process resume file to extract skills
def process_resume(user_id, file_stream, file_type):
    if file_type == 'application/pdf':
        text = extract_text_from_pdf(file_stream)
    else:
        raise ValueError("Unsupported file type")
    
    keywords = extract_skills_from_text(text)
    
    return {"userId": user_id, "keywords": keywords}

# Modified recommend_jobs function
def recommend_jobs(user_id, jobs, csv_file):
    # Load the CSV file containing user skills
    df = pd.read_csv(csv_file, header=None)

    # Get the user's skills from the CSV file
    user_skills = []
    for index, row in df.iterrows():
        if row[0] == user_id:
            user_skills = row[1:].tolist()
            break

    # Preprocess the user's skills
    user_skills_processed = preprocess_skills(' '.join(user_skills))

    # Transform the user's skills into a TF-IDF vector
    user_vector = vectorizer.transform([user_skills_processed])

    # Calculate cosine similarity with all jobs
    sim_scores = []
    for job in jobs:
        # Preprocess job skills
        job_skills = preprocess_skills(' '.join(extract_skills_from_text(job['skills'])))
        
        # Transform job skills into a TF-IDF vector
        job_vector = vectorizer.transform([job_skills])

        # Calculate cosine similarity
        similarity = cosine_similarity(user_vector, job_vector).flatten()[0]
        print(similarity)
        sim_scores.append(similarity)

    # Get indices of jobs sorted by similarity scores (descending)
    job_indices = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)

    # Prepare structured output
    recommended_jobs = []
    for idx in job_indices:
        job = jobs[idx]
        recommended_jobs.append(job)

    return recommended_jobs

@app.route('/api', methods=['GET'])
def home():
    return 'Welcome to the Flask Service!'

@app.route('/api/extract_keywords', methods=['POST'])
def extract_keywords_endpoint():
    user_id = request.form['userId']
    resume_file = request.files['resume']
    file_type = resume_file.mimetype

    # Process the resume file
    result = process_resume(user_id, resume_file.stream, file_type)

    # Path to the CSV file
    csv_file_path = 'api/user_skills/keywords.csv'
    
    # Read existing data
    existing_data = []
    user_found = False
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0] == user_id:
                existing_data.append([user_id] + result['keywords'])
                user_found = True
            else:
                existing_data.append(row)
    
    # Add new data if user ID not found
    if not user_found:
        existing_data.append([user_id] + result['keywords'])
    
    # Write updated data back to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(existing_data)

    return jsonify({'message': 'Keywords extracted and updated successfully'})

@app.route('/api/recommend_jobs', methods=['POST'])
def recommend_jobs_api():
    try:
        user_id = request.json['user_id']
        jobs = request.json['jobs']
        csv_file = 'api/user_skills/keywords.csv'

        recommended_jobs = recommend_jobs(user_id, jobs, csv_file)

        return jsonify(recommended_jobs)
    except Exception as e:
        print(e)
        return jsonify([])

if __name__ == "__main__":
    app.run(debug=True)
