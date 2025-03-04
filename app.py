from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

# Function to rank resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# API endpoint to process resumes
@app.route('/process_resumes', methods=['POST'])
def process_resumes():
    if 'job_description' not in request.form:
        return jsonify({"error": "Job description is required"}), 400
    
    if 'resumes' not in request.files:
        return jsonify({"error": "No resumes uploaded"}), 400

    job_description = request.form['job_description']
    uploaded_files = request.files.getlist('resumes')

    resumes = []
    resume_names = []

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
        resume_names.append(file.filename)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Prepare response
    results = [{"resume": name, "score": float(score)} for name, score in zip(resume_names, scores)]
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return jsonify(results)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)