from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import base64
import cv2
import tempfile
import os
from src.features.text_features import extract_text_features
from src.features.image_features import extract_image_features
from src.features.fusion import fuse_features

app = Flask(__name__)
CORS(app)

# Charger le modèle et le vectoriseur
if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
else:
    model = None
    vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Modèle non chargé. Veuillez entraîner le modèle d\'abord.'}), 500

    data = request.get_json()
    text = data.get('text', '')
    image_b64 = data.get('image', '')
    
    if not text or not image_b64:
        return jsonify({'error': 'Texte et image requis'}), 400
    
    # Décoder base64
    try:
        image_bytes = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name
        img_vec = extract_image_features(temp_path)
        os.unlink(temp_path)
    except Exception as e:
        return jsonify({'error': f'Erreur image: {str(e)}'}), 400
    
    # Features texte
    tfidf, extra, _ = extract_text_features([text], vectorizer=vectorizer)
    fused = fuse_features(tfidf[0], extra[0], img_vec)
    
    proba = model.predict_proba([fused])[0]
    pred = model.predict([fused])[0]
    
    return jsonify({
        'prediction': 'coherent' if pred == 1 else 'incoherent',
        'confidence': float(max(proba)),
        'probabilities': {'coherent': float(proba[1]), 'incoherent': float(proba[0])}
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
