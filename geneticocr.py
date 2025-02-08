import io
import json
import os
import re

import fitz
import pytesseract
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

GENETIC_PARAMETERS = [
    "Genetic Name", "DNA Type", "Disease Risk Assessment",
    "Pharmacogenomics", "Ancestry Composition", "Polygenic Risk Score",
    "Mitochondrial DNA Analysis", "HLA Typing", "Inherited Traits",
    "Epigenetic Modifications"
]

def extract_genetic_values(raw_text):
    extracted_data = {}
    raw_text = raw_text.replace("\n", " ")  

    for i in range(len(GENETIC_PARAMETERS)):
        param = GENETIC_PARAMETERS[i]
        next_param = GENETIC_PARAMETERS[i + 1] if i + 1 < len(GENETIC_PARAMETERS) else None

        if next_param:
            pattern = rf"{re.escape(param)}[:\s]*([^\n]+?)(?={re.escape(next_param)}|$)"
        else:
            pattern = rf"{re.escape(param)}[:\s]*([^\n]+)"

        match = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)

        if match:
          
            value = match.group(1).strip()
            extracted_data[param] = value
        else:

            extracted_data[param] = "Not Found"

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    extracted_text = extract_text_from_pdf(file_path)
    if not extracted_text:
        extracted_text = extract_text_from_images(file_path)

    # Extract genetic values from text
    extracted_data = extract_genetic_values(extracted_text)

    return jsonify({
        "filename": filename,
        "extracted_data": extracted_data  # Display extracted data
    })

if __name__ == "__main__":
    app.run(debug=True)
