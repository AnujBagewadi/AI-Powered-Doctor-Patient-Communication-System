from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import uuid

# Conditionally import AI-related libraries with proper error handling
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import gtts
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure static directories exist
os.makedirs(os.path.join('static', 'audio'), exist_ok=True)

# Initialize SQLAlchemy with app (proper Flask 2.x setup)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize AI models only if transformers is available
summarizer = None
translator = None
qa_model = None

if TRANSFORMERS_AVAILABLE:
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        translator = pipeline("translation_en_to_fr", model="t5-small")
        qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    except Exception as e:
        print(f"Error loading AI models: {str(e)}")

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_doctor = db.Column(db.Boolean, default=False)
    medical_records = db.relationship('MedicalRecord', backref='patient', lazy=True)
    discussions = db.relationship('Discussion', backref='author', lazy=True)

class MedicalRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    record_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    diagnosis = db.Column(db.Text, nullable=False)
    prescription = db.Column(db.Text, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    summary = db.Column(db.Text, nullable=True)

class Discussion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    comments = db.relationship('Comment', backref='discussion', lazy=True, cascade='all, delete-orphan')

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    discussion_id = db.Column(db.Integer, db.ForeignKey('discussion.id'), nullable=False)
    author = db.relationship('User', backref='comments')

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        is_doctor = True if request.form.get('is_doctor') else False
        
        user = User.query.filter_by(email=email).first()
        if user:
            return "Email already registered."
        
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password, method='sha256'),
            is_doctor=is_doctor
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            return "Please check your login details and try again."
        
        login_user(user)
        return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_doctor:
        return render_template('doctor_dashboard.html')
    return render_template('patient_dashboard.html')

# Medical Report Summarizer
@app.route('/summarize', methods=['POST'])
@login_required
def summarize_report():
    if not TRANSFORMERS_AVAILABLE or summarizer is None:
        return jsonify({'error': 'Summarization functionality is not available'}), 503
    
    report_text = request.form.get('report_text')
    if not report_text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        summary = summarizer(report_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        
        if 'record_id' in request.form:
            record_id = request.form.get('record_id')
            record = MedicalRecord.query.get(record_id)
            if record:
                record.summary = summary
                db.session.commit()
        
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': f'Summarization error: {str(e)}'}), 500

# AI Chatbot
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    if not TRANSFORMERS_AVAILABLE or qa_model is None:
        return jsonify({'error': 'Chat functionality is not available'}), 503
    
    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    context = "I am a medical AI assistant. I can help with general medical questions, but always consult a doctor for specific medical advice."
    
    try:
        answer = qa_model(question=query, context=context)['answer']
        
        # Save chat history
        chat_entry = ChatHistory(
            user_id=current_user.id,
            message=query,
            response=answer
        )
        db.session.add(chat_entry)
        db.session.commit()
        
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'}), 500

# Discussion Forum
@app.route('/discussions')
@login_required
def discussions():
    all_discussions = Discussion.query.order_by(Discussion.date_posted.desc()).all()
    return render_template('discussions.html', discussions=all_discussions)

@app.route('/discussion/new', methods=['GET', 'POST'])
@login_required
def new_discussion():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        
        if not title or not content:
            return "Title and content are required", 400
        
        discussion = Discussion(title=title, content=content, user_id=current_user.id)
        db.session.add(discussion)
        db.session.commit()
        
        return redirect(url_for('discussions'))
    
    return render_template('create_discussion.html')

@app.route('/discussion/<int:discussion_id>')
@login_required
def view_discussion(discussion_id):
    discussion = Discussion.query.get_or_404(discussion_id)
    return render_template('view_discussion.html', discussion=discussion)

@app.route('/discussion/<int:discussion_id>/comment', methods=['POST'])
@login_required
def add_comment(discussion_id):
    content = request.form.get('content')
    
    if not content:
        return "Comment content is required", 400
    
    # Verify discussion exists
    discussion = Discussion.query.get_or_404(discussion_id)
    
    comment = Comment(content=content, user_id=current_user.id, discussion_id=discussion_id)
    db.session.add(comment)
    db.session.commit()
    
    return redirect(url_for('view_discussion', discussion_id=discussion_id))

# Medical History Tracker
@app.route('/medical_history')
@login_required
def medical_history():
    if current_user.is_doctor:
        records = MedicalRecord.query.join(User).all()
    else:
        records = MedicalRecord.query.filter_by(user_id=current_user.id).all()
    
    return render_template('medical_history.html', records=records)

@app.route('/add_record', methods=['GET', 'POST'])
@login_required
def add_record():
    if not current_user.is_doctor:
        return "Only doctors can add medical records", 403
    
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        diagnosis = request.form.get('diagnosis')
        prescription = request.form.get('prescription')
        notes = request.form.get('notes')
        
        if not user_id or not diagnosis:
            return "User ID and diagnosis are required", 400
        
        # Validate user exists
        patient = User.query.get_or_404(user_id)
        
        # Auto-summarize the notes if summarizer is available
        summary = None
        if notes and TRANSFORMERS_AVAILABLE and summarizer is not None:
            try:
                summary = summarizer(notes, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            except Exception as e:
                print(f"Summarization error: {str(e)}")
        
        record = MedicalRecord(
            user_id=user_id,
            diagnosis=diagnosis,
            prescription=prescription,
            notes=notes,
            summary=summary
        )
        
        db.session.add(record)
        db.session.commit()
        
        return redirect(url_for('medical_history'))
    
    patients = User.query.filter_by(is_doctor=False).all()
    return render_template('add_record.html', patients=patients)

# Language Translation
@app.route('/translate', methods=['POST'])
@login_required
def translate_text():
    if not TRANSFORMERS_AVAILABLE or translator is None:
        return jsonify({'error': 'Translation functionality is not available'}), 503
    
    text = request.form.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    target_lang = request.form.get('target_lang', 'fr')  # Default to French
    
    try:
        # This is a simplified example - in production, you'd want to handle multiple languages
        translated = translator(text)[0]['translation_text']
        
        return jsonify({'translated_text': translated})
    except Exception as e:
        return jsonify({'error': f'Translation error: {str(e)}'}), 500

# Text-to-Speech
@app.route('/text_to_speech', methods=['POST'])
@login_required
def text_to_speech():
    if not GTTS_AVAILABLE:
        return jsonify({'error': 'Text-to-speech functionality is not available'}), 503
    
    text = request.form.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    language = request.form.get('language', 'en')
    
    try:
        tts = gtts.gTTS(text=text, lang=language, slow=False)
        
        # Generate a unique filename
        filename = f"speech_{uuid.uuid4()}.mp3"
        file_path = os.path.join('static', 'audio', filename)
        
        # Save the audio file
        tts.save(file_path)
        
        return jsonify({'audio_url': url_for('static', filename=f'audio/{filename}')})
    except Exception as e:
        return jsonify({'error': f'Text-to-speech error: {str(e)}'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)