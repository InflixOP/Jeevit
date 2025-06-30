# Jeevit 🧬💊

**A comprehensive health platform that offers users assistance with medical reports and dosage recommendations based on genetic data while also providing a personalized exercise trainer.**


## 🌟 Features

### 🧬 Genetic Analysis
- **Personalized Medicine**: Analyze genetic data to provide tailored medical insights
- **Dosage Recommendations**: AI-powered medication dosage suggestions based on genetic markers
- **Pharmacogenomics**: Understanding how genes affect drug response

### 📋 Medical Report Analysis
- **Smart Report Processing**: Computer vision-powered medical report analysis
- **Data Extraction**: Automatically extract key information from medical documents
- **Trend Analysis**: Track health metrics over time

### 💪 Personal Exercise Trainer
- **AI Fitness Coach**: Personalized workout recommendations
- **Form Analysis**: Computer vision for exercise form correction
- **Progress Tracking**: Monitor fitness goals and achievements

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, TensorFlow/PyTorch
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Genetic Analysis**: BioPython, custom algorithms
- **Frontend**: HTML5, CSS3, JavaScript

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip
virtualenv (recommended)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/jeevit.git
cd jeevit
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize database**
```bash
python app.py init-db
```

6. **Run the application**
```bash
python app.py
```

Visit `http://localhost:5000` to access Jeevit!



## 🔬 API Endpoints

### Genetic Analysis
- `POST /api/genetic/analyze` - Analyze genetic data
- `POST /api/genetic/dosage` - Get dosage recommendations

### Medical Reports
- `POST /api/reports/upload` - Upload medical report
- `GET /api/reports/analysis/{id}` - Get report analysis

### Exercise Trainer
- `POST /api/exercise/recommend` - Get exercise recommendations
- `POST /api/exercise/analyze-form` - Analyze exercise form

## 🧪 Usage Examples

### Genetic Analysis
```python
import requests

# Upload genetic data
response = requests.post('http://localhost:5000/api/genetic/analyze', 
                        files={'genetic_file': open('sample.vcf', 'rb')})
print(response.json())
```

### Medical Report Processing
```python
# Upload medical report image
response = requests.post('http://localhost:5000/api/reports/upload',
                        files={'report': open('medical_report.pdf', 'rb')})
analysis = response.json()
```


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

