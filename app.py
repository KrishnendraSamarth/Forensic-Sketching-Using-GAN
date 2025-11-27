"""
Sketch2FaceGAN Flask Application
A web app that converts sketch images to realistic faces using GAN
"""

from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
import io
from model.model_loader import generate_face

app = Flask(__name__)
CORS(app)  # Enable CORS for local testing
app.config['DEBUG'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Handle sketch upload and generate face"""
    try:
        # Check if file is present in request
        if 'sketch' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['sketch']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is an image
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Reset file pointer to beginning (in case it was read before)
        file.seek(0)
        
        # Generate face from sketch
        generated_image_bytes = generate_face(file)
        
        # Return generated image
        return send_file(
            io.BytesIO(generated_image_bytes),
            mimetype='image/png',
            as_attachment=False
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

