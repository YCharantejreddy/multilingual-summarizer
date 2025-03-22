from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from youtube_transcript_api import YouTubeTranscriptApi
import PyPDF2
import re
import os
import logging
from datetime import datetime
from rouge_score import rouge_scorer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data for sentence tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')  # Use env var for Azure
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file size limit
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///summaries.db')  # Use Azure's DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename='app.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize database and bcrypt
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for authentication
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Summary model for storing summaries
class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# Create database tables
try:
    with app.app_context():
        db.create_all()
    logging.info("Database tables created successfully")
except Exception as e:
    logging.error(f"Error creating database tables: {str(e)}")

# Load the model and tokenizer directly from Hugging Face
model_name = "google/pegasus-xsum"  # Switched to a smaller model
tokenizer = None
model = None
try:
    logging.info(f"Loading model {model_name} directly from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/hf-cache")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="/tmp/hf-cache")
    model.eval()
    logging.info("Model and tokenizer loaded successfully")
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {str(e)}")
    print(f"Warning: Could not load model {model_name}. Summarization will not work until the model is loaded.")

# Load user for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Function to preprocess text (remove special tokens like <extra_id_0>)
def preprocess_text(text):
    text = re.sub(r'<extra_id_\d+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to chunk long text
def chunk_text(text, max_length=512):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence)) if tokenizer else len(sentence.split())
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        return text, None
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return None, str(e)

# Function to extract transcript from YouTube URL
def extract_transcript(youtube_url):
    try:
        video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", youtube_url)
        if not video_id:
            return None, "Invalid YouTube URL"
        video_id = video_id.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return text, None
    except Exception as e:
        logging.error(f"Error extracting transcript from YouTube: {str(e)}")
        return None, str(e)

# Function to calculate ROUGE scores
def calculate_rouge(original_text, summary):
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(original_text, summary)
        return scores
    except Exception as e:
        logging.error(f"Error calculating ROUGE scores: {str(e)}")
        return None

# Function to summarize text with improved parameters
def summarize_text(text, language="en_XX", max_length=50, min_length=10, num_beams=1, length_penalty=1.0):
    if not model or not tokenizer:
        return None, "Model not loaded. Summarization is unavailable."
    try:
        text = preprocess_text(text)
        chunks = chunk_text(text, max_length=512)
        summaries = []
        for chunk in chunks:
            inputs = tokenizer(
                f"summarize: {chunk}",
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length // len(chunks),
                min_length=min_length // len(chunks),
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.9,
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            summaries.append(summary)
        final_summary = " ".join(summaries)
        final_summary = preprocess_text(final_summary)
        return final_summary, None
    except Exception as e:
        logging.error(f"Error summarizing text: {str(e)}")
        return None, str(e)

# Route for the homepage
@app.route('/', methods=['GET'])
def index():
    logging.info("Rendering index page")
    return render_template('index.html')

# Route for login
@app.route('/login', methods=['GET', 'POST'])
def login():
    logging.info("Accessing login route")
    if current_user.is_authenticated:
        logging.info("User already authenticated, redirecting to index")
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        logging.info(f"Login attempt for username: {username}")
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            logging.info(f"User {username} logged in successfully")
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
            logging.warning(f"Failed login attempt for username: {username}")
    try:
        logging.info("Rendering login.html")
        return render_template('login.html')
    except Exception as e:
        logging.error(f"Error rendering login.html: {str(e)}")
        return "Error rendering login page", 500

# Route for registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    logging.info("Accessing register route")
    if current_user.is_authenticated:
        logging.info("User already authenticated, redirecting to index")
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        logging.info(f"Registration attempt for username: {username}")
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            logging.warning(f"Username {username} already exists")
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            logging.info(f"User {username} registered successfully")
            return redirect(url_for('login'))
    try:
        logging.info("Rendering register.html")
        return render_template('register.html')
    except Exception as e:
        logging.error(f"Error rendering register.html: {str(e)}")
        return "Error rendering register page", 500

# Route for logout
@app.route('/logout')
@login_required
def logout():
    logging.info(f"User {current_user.username} logging out")
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

# Route for summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    input_type = request.form.get('input_type')
    language = request.form.get('language', 'en_XX')
    text = None
    error = None

    if input_type == 'text':
        text = request.form.get('text')
        if not text:
            error = "Text input cannot be empty."
    elif input_type == 'youtube':
        youtube_url = request.form.get('youtube_url')
        if not youtube_url:
            error = "YouTube URL cannot be empty."
        else:
            text, error = extract_transcript(youtube_url)
    elif input_type == 'pdf':
        if 'pdf_file' not in request.files:
            error = "No PDF file uploaded."
        else:
            pdf_file = request.files['pdf_file']
            if pdf_file.filename == '':
                error = "No PDF file selected."
            else:
                text, error = extract_text_from_pdf(pdf_file)

    if error or not text:
        return render_template('index.html', error=error or "Unable to extract text.")

    summary, error = summarize_text(text, language=language)
    if error:
        return render_template('index.html', error=f"Error during summarization: {error}")

    # Calculate ROUGE scores
    rouge_scores = calculate_rouge(text, summary)

    # Save summary to database if user is logged in
    if current_user.is_authenticated:
        new_summary = Summary(
            user_id=current_user.id,
            original_text=text,
            summary=summary,
            language=language
        )
        db.session.add(new_summary)
        db.session.commit()

    return render_template('index.html', summary=summary, rouge_scores=rouge_scores)

# Route to view saved summaries
@app.route('/my_summaries')
@login_required
def my_summaries():
    summaries = Summary.query.filter_by(user_id=current_user.id).all()
    logging.info("Rendering my_summaries page")
    return render_template('my_summaries.html', summaries=summaries)

# Route to export summary as PDF
@app.route('/export_summary/<int:summary_id>')
@login_required
def export_summary(summary_id):
    summary = Summary.query.get_or_404(summary_id)
    if summary.user_id != current_user.id:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('index'))

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "Summary Report")
    p.drawString(100, 730, f"Language: {summary.language}")
    p.drawString(100, 710, f"Created At: {summary.created_at}")
    text_object = p.beginText(100, 690)
    text_object.setFont("Helvetica", 12)
    for line in summary.summary.split('\n'):
        text_object.textLine(line)
    p.drawText(text_object)
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"summary_{summary_id}.pdf",
        mimetype="application/pdf"
    )

# API endpoint for summarization
@app.route('/api/summarize', methods=['POST'])
@login_required
def api_summarize():
    data = request.json
    input_type = data.get("input_type")
    language = data.get("language", "en_XX")
    text = data.get("text")
    if input_type == "text" and text:
        summary, error = summarize_text(text, language)
        if error:
            return jsonify({"error": error}), 500
        rouge_scores = calculate_rouge(text, summary)
        return jsonify({"summary": summary, "rouge_scores": rouge_scores})
    return jsonify({"error": "Invalid input"}), 400

# Route for the About page
@app.route('/about')
def about():
    logging.info("Rendering about page")
    return render_template('about.html')

# Route for the Test Cases page
@app.route('/testcases')
def testcases():
    logging.info("Rendering testcases page")
    return render_template('testcases.html')

# Route for the Analyze page
@app.route('/analyze')
def analyze():
    logging.info("Rendering analyze page")
    return render_template('analyze.html')

if __name__ == "__main__":
    try:
        print("Starting Flask server...")
        port = int(os.getenv('PORT', 5000))  # Use Azure's PORT env var
        app.run(debug=False, host='0.0.0.0', port=port)
        print(f"Server is running on port {port}")
    except Exception as e:
        print(f"Error starting Flask server: {str(e)}")
        logging.error(f"Error starting Flask server: {str(e)}")