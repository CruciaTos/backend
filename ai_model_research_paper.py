import os
import sqlite3
import numpy as np
import io
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import imagehash
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import cv2
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import logging
import sys
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plagiarism_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArXivPlagiarismDetector:
    def __init__(self, db_path='arxiv_db.sqlite', index_dir='vector_indices'):
        self.db_path = db_path
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        # Initialize with progress bar
        with tqdm(total=4, desc="Initializing System") as pbar:
            pbar.set_description("Loading SPECTER model")
            self.text_model = SentenceTransformer('allenai-specter')
            pbar.update(1)
            
            pbar.set_description("Initializing TF-IDF")
            self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            pbar.update(1)
            
            pbar.set_description("Setting up database")
            self._init_db()
            pbar.update(1)
            
            pbar.set_description("Loading search indices")
            self._load_or_create_indices()
            pbar.update(1)

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY,
                    arxiv_id TEXT,
                    file_hash TEXT UNIQUE,
                    file_name TEXT,
                    file_path TEXT,
                    title TEXT,
                    authors TEXT,
                    published_date TEXT,
                    content TEXT,
                    abstract TEXT,
                    image_hash TEXT,
                    processed_at TEXT,
                    error_message TEXT,
                    file_size INTEGER
                )
            ''')
            
            # Check and add missing columns
            cursor.execute("PRAGMA table_info(papers)")
            columns = {column[1] for column in cursor.fetchall()}
            
            if 'file_size' not in columns:
                cursor.execute('ALTER TABLE papers ADD COLUMN file_size INTEGER')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_arxiv_id ON papers (arxiv_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON papers (file_hash)')
            conn.commit()

    def _load_or_create_indices(self):
        self.text_index_path = os.path.join(self.index_dir, 'text_index.faiss')
        if os.path.exists(self.text_index_path):
            with tqdm(total=2, desc="Loading text index", leave=False) as pbar:
                self.text_index = faiss.read_index(self.text_index_path)
                pbar.update(1)
                with open(os.path.join(self.index_dir, 'text_ids.pkl'), 'rb') as f:
                    self.text_index_ids = pickle.load(f)
                pbar.update(1)
        else:
            self.text_index = None
            self.text_index_ids = []

    def _validate_file(self, file_path):
        """Validate file before processing"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "Empty file (0KB)", file_size
            
            if not file_path.lower().endswith('.pdf'):
                return True, "", file_size
                
            # Check PDF header
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    return False, "Not a valid PDF file", file_size
            
            return True, "", file_size
        except Exception as e:
            return False, f"File access error: {str(e)}", 0

    def _extract_arxiv_metadata(self, file_path):
        filename = os.path.basename(file_path)
        match = re.match(r'^(\d{4}\.\d{4,5})(v\d+)?\.pdf$', filename)
        if match:
            return {
                'arxiv_id': match.group(1),
                'version': match.group(2)[1:] if match.group(2) else None
            }
        return None

    def _extract_paper_metadata(self, content, filename):
        lines = content.split('\n')
        metadata = {
            'title': filename.replace('.pdf', '')[:200],
            'authors': 'Unknown',
            'abstract': ''
        }
        
        # Find title (first non-empty line)
        for line in lines:
            if line.strip() and len(line.strip()) > 10:
                metadata['title'] = line.strip()[:500]
                break
                
        # Find authors (line after title)
        for i, line in enumerate(lines):
            if line.strip() == metadata['title'] and i+1 < len(lines):
                metadata['authors'] = lines[i+1].strip()[:300]
                break
                
        # Find abstract
        abstract_pos = content.lower().find('abstract')
        if abstract_pos >= 0:
            abstract = content[abstract_pos+7:abstract_pos+2000].strip()
            metadata['abstract'] = abstract.split('\n\n')[0][:1000]
            
        return metadata

    def _file_hash(self, file_path):
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_image_hash(self, file_path):
        """Generate image hash if possible"""
        try:
            if not file_path.lower().endswith('.pdf'):
                with Image.open(file_path) as img:
                    return str(imagehash.phash(img))
            
            # For PDFs, try to extract first image
            try:
                reader = PdfReader(file_path)
                page = reader.pages[0]
                for image in page.images:
                    try:
                        img = Image.open(io.BytesIO(image.data))
                        return str(imagehash.phash(img))
                    except Exception:
                        continue
            except Exception:
                pass
            
            return None
        except Exception:
            return None

    def _extract_text(self, file_path):
        """Robust text extraction with fallbacks"""
        try:
            # First try normal PDF extraction
            if file_path.lower().endswith('.pdf'):
                try:
                    reader = PdfReader(file_path)
                    text = '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])
                    
                    # Extract abstract if we got good text
                    if len(text.strip()) > 100:
                        abstract = ''
                        for page in reader.pages[:2]:
                            page_text = page.extract_text() or ''
                            if 'abstract' in page_text.lower():
                                abstract = page_text.split('abstract', 1)[1].split('\n\n', 1)[0]
                                break
                        return text, abstract, ""
                        
                    # Fallback to OCR if little text
                    logger.info(f"Using OCR for {os.path.basename(file_path)} (little text)")
                except Exception as e:
                    logger.info(f"PDF extraction failed, using OCR: {str(e)}")
            
            # OCR fallback
            try:
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img)
                return text, "", ""
            except Exception as e:
                return "", "", f"OCR failed: {str(e)}"
                
        except Exception as e:
            return "", "", f"Extraction failed: {str(e)}"

    def add_paper(self, file_path):
        # Validate file first
        is_valid, valid_msg, file_size = self._validate_file(file_path)
        if not is_valid:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO papers 
                    (file_hash, file_name, file_path, error_message, processed_at, file_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    self._file_hash(file_path),
                    os.path.basename(file_path),
                    file_path,
                    valid_msg,
                    datetime.now().isoformat(),
                    file_size
                ))
                conn.commit()
            return False, valid_msg

        file_hash = self._file_hash(file_path)
        arxiv_meta = self._extract_arxiv_metadata(file_path)
        
        # Check if paper exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM papers WHERE file_hash = ?', (file_hash,))
            if cursor.fetchone() is not None:
                return True, "Already exists"

        # Process document
        try:
            with tqdm(total=4, desc=f"Processing {os.path.basename(file_path)}", leave=False) as pbar:
                pbar.set_description("Extracting text")
                content, abstract, extract_error = self._extract_text(file_path)
                if extract_error:
                    raise Exception(extract_error)
                pbar.update(1)
                
                pbar.set_description("Extracting metadata")
                paper_meta = self._extract_paper_metadata(content, os.path.basename(file_path))
                pbar.update(1)
                
                pbar.set_description("Generating features")
                image_hash = self._get_image_hash(file_path)
                text_embedding = self.text_model.encode(content)
                pbar.update(1)
                
                pbar.set_description("Saving to database")
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO papers 
                        (arxiv_id, file_hash, file_name, file_path, title, authors, 
                         published_date, content, abstract, image_hash, processed_at, file_size)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        arxiv_meta['arxiv_id'] if arxiv_meta else None,
                        file_hash,
                        os.path.basename(file_path),
                        file_path,
                        paper_meta['title'],
                        paper_meta['authors'],
                        arxiv_meta['version'] if arxiv_meta else None,
                        content,
                        abstract,
                        image_hash,
                        datetime.now().isoformat(),
                        file_size
                    ))
                    paper_id = cursor.lastrowid
                    conn.commit()
                pbar.update(1)

            # Update indices
            self._update_indices(paper_id, text_embedding)
            return True, "Success"
            
        except Exception as e:
            # Record failed attempt
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO papers 
                    (file_hash, file_name, file_path, error_message, processed_at, file_size)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    file_hash,
                    os.path.basename(file_path),
                    file_path,
                    str(e),
                    datetime.now().isoformat(),
                    file_size
                ))
                conn.commit()
            return False, str(e)

    def _update_indices(self, paper_id, text_embedding):
        if self.text_index is None:
            self.text_index = faiss.IndexFlatIP(text_embedding.shape[0])
            self.text_index_ids = []
        
        text_embedding = text_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(text_embedding)
        self.text_index.add(text_embedding)
        self.text_index_ids.append(paper_id)
        
        # Save indices
        faiss.write_index(self.text_index, self.text_index_path)
        with open(os.path.join(self.index_dir, 'text_ids.pkl'), 'wb') as f:
            pickle.dump(self.text_index_ids, f)

    def batch_add_papers(self, folder_path, num_workers=4):
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found - {folder_path}")
            return False
            
        # Find PDF files
        pdf_files = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith('.pdf') and not f.startswith('~$'):
                    pdf_files.append(os.path.join(root, f))
        
        if not pdf_files:
            logger.error(f"No PDF files found in {folder_path}")
            return False
            
        logger.info(f"Found {len(pdf_files)} PDFs to process")

        # Process files
        success_count = 0
        with tqdm(total=len(pdf_files), desc="Processing papers") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for file_path in pdf_files:
                    future = executor.submit(self.add_paper, file_path)
                    future.add_done_callback(lambda _: pbar.update(1))
                    futures.append(future)
                
                for future in futures:
                    try:
                        success, message = future.result()
                        if not success:
                            logger.warning(f"Processing warning: {message}")
                        else:
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Error processing file: {str(e)}")

        logger.info(f"Processed {len(pdf_files)} files. Successfully added {success_count} papers.")
        return success_count > 0

    def search_similar(self, query_file_path, k=5, threshold=0.6):
        if not os.path.exists(query_file_path):
            logger.error(f"File not found - {query_file_path}")
            return []
            
        # Extract query features
        with tqdm(total=3, desc="Analyzing query") as pbar:
            pbar.set_description("Extracting text")
            content, _, _ = self._extract_text(query_file_path)
            if not content.strip():
                logger.error("Could not extract text from query file")
                return []
            pbar.update(1)
            
            pbar.set_description("Generating embedding")
            query_embedding = self.text_model.encode(content).astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            pbar.update(1)
            
            pbar.set_description("Searching database")
            distances, indices = self.text_index.search(query_embedding, k)
            pbar.update(1)
        
        # Get results
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            for distance, idx in zip(distances[0], indices[0]):
                if idx >= 0 and distance >= threshold:
                    paper_id = self.text_index_ids[idx]
                    cursor.execute('''
                        SELECT id, arxiv_id, file_name, title, authors, abstract 
                        FROM papers WHERE id = ? AND error_message IS NULL
                    ''', (paper_id,))
                    doc = cursor.fetchone()
                    
                    if doc:
                        results.append({
                            'paper_id': doc['id'],
                            'arxiv_id': doc['arxiv_id'],
                            'file_name': doc['file_name'],
                            'title': doc['title'],
                            'authors': doc['authors'],
                            'abstract': doc['abstract'],
                            'similarity_score': float(distance)
                        })

        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

    def get_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM papers WHERE error_message IS NULL')
            valid_papers = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM papers WHERE error_message IS NOT NULL')
            error_papers = cursor.fetchone()[0]
            
            cursor.execute('SELECT MIN(processed_at), MAX(processed_at) FROM papers')
            min_date, max_date = cursor.fetchone()
            
        return {
            'valid_papers': valid_papers,
            'error_papers': error_papers,
            'date_range': {
                'oldest': min_date,
                'newest': max_date
            },
            'index_size': len(self.text_index_ids) if self.text_index_ids else 0
        }

    def get_failed_papers(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT file_name, file_path, error_message, file_size
                FROM papers WHERE error_message IS NOT NULL
                ORDER BY file_size DESC
            ''')
            return cursor.fetchall()

    def save_model(self, model_path):
        """Save the complete model state to a directory"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save database
        db_copy_path = os.path.join(model_path, 'arxiv_db.sqlite')
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as src, open(db_copy_path, 'wb') as dst:
                dst.write(src.read())
        
        # Save indices
        index_copy_dir = os.path.join(model_path, 'vector_indices')
        os.makedirs(index_copy_dir, exist_ok=True)
        
        if os.path.exists(self.text_index_path):
            faiss.write_index(self.text_index, os.path.join(index_copy_dir, 'text_index.faiss'))
            with open(os.path.join(index_copy_dir, 'text_ids.pkl'), 'wb') as f:
                pickle.dump(self.text_index_ids, f)
        
        logger.info(f"Model saved successfully to {model_path}")

    @classmethod
    def load_model(cls, model_path):
        """Load a saved model from directory"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Check required files exist
        required_files = [
            os.path.join(model_path, 'arxiv_db.sqlite'),
            os.path.join(model_path, 'vector_indices', 'text_index.faiss'),
            os.path.join(model_path, 'vector_indices', 'text_ids.pkl')
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required model file not found: {file_path}")
        
        # Create detector instance
        detector = cls(
            db_path=os.path.join(model_path, 'arxiv_db.sqlite'),
            index_dir=os.path.join(model_path, 'vector_indices')
        )
        
        logger.info(f"Model loaded successfully from {model_path}")
        return detector

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models
logger.info("Loading Whisper model...")
whisper_model = whisper.load_model("base")
logger.info("Whisper model loaded successfully!")

logger.info("Initializing Plagiarism Detector...")
try:
    plagiarism_detector = ArXivPlagiarismDetector(
        db_path='arxiv_db.sqlite',
        index_dir='vector_indices'
    )
    logger.info("Plagiarism Detector ready!")
except Exception as e:
    logger.error(f"Failed to initialize detector: {str(e)}", exc_info=True)
    raise

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            logger.warning("No audio file provided in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.warning("Empty audio file uploaded")
            return jsonify({'error': 'Empty file uploaded'}), 400

        # Create temp file safely
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)
            logger.info(f"Saved temporary audio file at {tmp_path}")

        try:
            logger.info("Starting transcription")
            result = whisper_model.transcribe(tmp_path)
            logger.info("Transcription completed successfully")
            return jsonify({
                'transcription': result['text'].strip(),
                'language': result['language']
            })
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info("Removed temporary audio file")

    except Exception as e:
        logger.error(f"Server error in transcription: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/detect_plagiarism', methods=['POST'])
def detect_plagiarism():
    try:
        # Check if request contains file or text
        if 'file' not in request.files and 'text' not in request.form:
            logger.error("No file or text provided in request")
            return jsonify({'error': 'Please provide either a file or text to analyze'}), 400

        results = []
        tmp_path = None
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                logger.error("Empty file uploaded")
                return jsonify({'error': 'Empty file uploaded'}), 400

            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp_path = tmp.name
                file.save(tmp_path)
                logger.info(f"Saved temporary file at {tmp_path}")

            try:
                logger.info("Starting plagiarism detection for file")
                results = plagiarism_detector.search_similar(tmp_path)
                logger.info(f"Found {len(results)} potential matches")
            except Exception as e:
                logger.error(f"Error during plagiarism detection: {str(e)}", exc_info=True)
                return jsonify({'error': f'Detection failed: {str(e)}'}), 500
        
        # Handle text input
        elif 'text' in request.form:
            text = request.form['text']
            if not text.strip():
                logger.error("Empty text provided")
                return jsonify({'error': 'Please provide some text to analyze'}), 400
            
            try:
                logger.info("Processing text for plagiarism detection")
                # Create temp file with the text
                with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w+') as tmp:
                    tmp_path = tmp.name
                    tmp.write(text)
                    tmp.flush()
                    logger.info(f"Saved temporary text file at {tmp_path}")
                
                results = plagiarism_detector.search_similar(tmp_path)
                logger.info(f"Found {len(results)} potential matches for text input")
            except Exception as e:
                logger.error(f"Error processing text: {str(e)}", exc_info=True)
                return jsonify({'error': f'Text processing failed: {str(e)}'}), 500

        # Format results
        formatted_results = []
        for result in results[:5]:  # Return top 5 matches
            formatted_results.append({
                'title': result.get('title', ''),
                'authors': result.get('authors', ''),
                'similarity_score': result.get('similarity_score', 0),
                'abstract': result.get('abstract', '')[:200] + '...' if result.get('abstract') else '',
                'file_name': result.get('file_name', '')
            })
        
        return jsonify({
            'status': 'success',
            'results': formatted_results
        })
            
    except Exception as e:
        logger.error(f"Unexpected server error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("Removed temporary file")

@app.route('/health', methods=['GET'])
def health_check():
    try:
        stats = plagiarism_detector.get_stats()
        return jsonify({
            'status': 'healthy',
            'services': {
                'whisper': 'active',
                'plagiarism_detector': 'active'
            },
            'database_stats': stats
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/add_papers', methods=['POST'])
def add_papers_endpoint():
    try:
        if 'folder_path' not in request.json:
            logger.error("No folder path provided in request")
            return jsonify({'error': 'No folder path provided'}), 400
            
        folder_path = request.json['folder_path']
        if not os.path.exists(folder_path):
            logger.error(f"Folder path does not exist: {folder_path}")
            return jsonify({'error': 'Folder path does not exist'}), 400
            
        logger.info(f"Starting batch processing for folder: {folder_path}")
        success = plagiarism_detector.batch_add_papers(folder_path)
        
        return jsonify({
            'status': 'success' if success else 'partial_success',
            'message': 'Papers processed successfully' if success else 'Some papers failed to process'
        })
    except Exception as e:
        logger.error(f"Error adding papers: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)