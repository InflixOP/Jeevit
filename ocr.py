import io
import os
import re

import fitz
import pytesseract
from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MEDICAL_PARAMETERS = {
    "Haemoglobin (Hb)": r"([\d\.]+)\s*L\s*\n[\d\.]+\s*-\s*[\d\.]+\s*gm/dL",
    "Uric Acid": r"Uric Acid.*?\n.*?\n.*?([\d\.]+)",
    "Total T3 (Triiodothyronine)": r"Total T3.*?\n.*?([\d\.]+)",
    "Total T4 (Thyroxine)": r"Total T4.*?\n.*?([\d\.]+)",
    "TSH 3rd Generation": r"TSH 3rd Generation.*?\n.*?([\d\.]+)",
    "Blood Pressure (BP)": r"Blood Pressure.*?([\d\/]+)\s*mmHg",
    "Heart Rate (Pulse)": r"Heart Rate.*?([\d]+)\s*bpm",
    "Respiratory Rate": r"Respiratory Rate.*?([\d]+)\s*breaths/min",
    "Body Temperature": r"Body Temperature.*?([\d\.]+)\s*[°C°F]+",
    "Oxygen Saturation (SpO₂)": r"Oxygen Saturation.*?([\d\.]+)\s*%",
}

def extract_medical_values(text):
    extracted_data = {}
    for param, pattern in MEDICAL_PARAMETERS.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted_data[param] = match.group(1)
    return extracted_data

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def extract_text_from_images(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        text += pytesseract.image_to_string(img) + "\n"
    return text.strip()
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    extracted_text = extract_text_from_pdf(file_path)
    if not extracted_text:
        extracted_text = extract_text_from_images(file_path)

    extracted_data = extract_medical_values(extracted_text)

    return jsonify({"extracted_data": extracted_data})

if __name__ == '__main__':
    app.run(debug=True)
