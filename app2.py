from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import tempfile
import logging
import sys
from pathlib import Path

# Add the models directory to Python path
models_path = str(Path(__file__).parent / "assets" / "models")
sys.path.append(models_path)

# Now import your detector
try:
    from ai_model_research_paper import ArXivPlagiarismDetector
except ImportError:
    try:
        # Try alternative naming if needed
        from ai_model_research_paper import ArXivPlagiarismDetector as Detector
        ArXivPlagiarismDetector = Detector
    except ImportError as e:
        print(f"Import failed: {e}")
        raise

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully!")

print("Initializing Plagiarism Detector...")
try:
    plagiarism_detector = ArXivPlagiarismDetector(
        db_path='arxiv_db.sqlite',
        index_dir='vector_indices'
    )
    print("Plagiarism Detector ready!")
except Exception as e:
    print(f"Failed to initialize detector: {str(e)}")
    raise

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty file uploaded'}), 400

        # Create temp file safely
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        try:
            result = whisper_model.transcribe(tmp_path)
            os.remove(tmp_path)
            return jsonify({
                'transcription': result['text'].strip(),
                'language': result['language']
            })
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/detect_plagiarism', methods=['POST'])
def detect_plagiarism():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file uploaded'}), 400

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        try:
            results = plagiarism_detector.search_similar(tmp_path)
            os.remove(tmp_path)
            
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
            logger.error(f"Plagiarism detection error: {str(e)}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return jsonify({'error': f'Detection failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'services': {
            'whisper': 'active',
            'plagiarism_detector': 'active'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)