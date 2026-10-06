# 🎭 Emotion Text Classification

A comprehensive machine learning project for classifying text into six emotional categories using Natural Language Processing (NLP) techniques and multiple classification models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Emotion Categories](#emotion-categories)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Contributing](#contributing)


## 🎯 Overview

This project implements an end-to-end emotion classification pipeline that analyzes text data and predicts the underlying emotion. The system leverages traditional machine learning algorithms combined with advanced text preprocessing and feature extraction techniques.

## 😊 Emotion Categories

The model classifies text into **6 emotion categories**:

| Label | Emotion   | Description                        |
|-------|-----------|-------------------------------------|
| 0     | 😢 Sadness  | Expressions of grief, sorrow       |
| 1     | 😄 Joy      | Happiness, excitement, pleasure    |
| 2     | ❤️ Love     | Affection, care, romantic feelings |
| 3     | 😠 Anger    | Frustration, rage, annoyance       |
| 4     | 😨 Fear     | Anxiety, worry, terror             |
| 5     | 😲 Surprise | Astonishment, amazement            |

## 📁 Project Structure

```
Emotion_Classification/
│
├── Emotion_Text_Classification.ipynb   # Main Jupyter notebook with full analysis
├── README.md                           # Project documentation
│
└── data/
    ├── train.jsonl                     # Training dataset
    ├── test.jsonl                      # Test dataset
    └── validation.jsonl                # Validation dataset
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook / JupyterLab / VS Code

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sk-koirala/Emotion_Text_Classification.git
   cd Emotion_Text_Classification
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk
   ```

4. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('punkt_tab')
   ```

## 📊 Dataset

The dataset consists of text samples labeled with one of six emotions:

| Split       | Samples  |
|-------------|----------|
| Training    | 16,000   |
| Validation  | 2,000    |
| Test        | 2,000    |
| **Total**   | **20,000** |

### Data Format

Each file is in JSONL format with the following structure:
```json
{"text": "sample text here", "label": 0}
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Text Cleaning:** Removal of special characters, URLs, mentions, and HTML tags
- **Lowercasing:** Converting all text to lowercase
- **Tokenization:** Splitting text into individual words
- **Stop Words Removal:** Filtering out common English stop words
- **Lemmatization:** Reducing words to their base form using WordNet Lemmatizer

### 2. Feature Extraction
- **TF-IDF Vectorization:** 
  - Max features: 5,000
  - N-gram range: (1, 2) - Unigrams and Bigrams
  - Min document frequency: 2
  - Max document frequency: 80%
  - Sublinear TF scaling enabled

### 3. Exploratory Data Analysis (EDA)
- Class distribution analysis
- Text length statistics
- Word frequency analysis
- Word cloud visualizations

## 🤖 Models Implemented

### 1. Multinomial Naive Bayes
- Probabilistic classifier based on Bayes' theorem
- Well-suited for text classification tasks
- Fast training and prediction

### 2. Logistic Regression
- Linear model for classification
- L2 regularization applied
- Multi-class classification using One-vs-Rest strategy

### 3. Support Vector Machine (SVM)
- LinearSVC implementation for efficiency
- Effective in high-dimensional spaces
- Robust to overfitting

### 4. Ensemble Voting Classifier
- Combines predictions from multiple models
- Soft voting for probability-based decisions
- Improved robustness and accuracy

## 📈 Results

### Model Performance Comparison

| Model                    | Accuracy | Macro F1 | Weighted F1 |
|--------------------------|----------|----------|-------------|
| Multinomial Naive Bayes  | ~85%     | ~0.84    | ~0.85       |
| Logistic Regression      | ~89%     | ~0.88    | ~0.89       |
| Support Vector Machine   | ~88%     | ~0.87    | ~0.88       |
| Ensemble Classifier      | ~89%     | ~0.88    | ~0.89       |

*Note: Results may vary slightly based on random state and hyperparameters*

## 🚀 Usage

### Running the Notebook

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook Emotion_Text_Classification.ipynb
   ```

2. Run all cells sequentially to:
   - Load and preprocess data
   - Perform exploratory data analysis
   - Train and evaluate models
   - Generate visualizations

### Making Predictions

```python
# Example prediction with trained model
sample_text = "I am so happy today!"
cleaned_text = preprocess_text(sample_text)
features = tfidf_vectorizer.transform([cleaned_text])
prediction = model.predict(features)
emotion = emotion_labels[prediction[0]]
print(f"Predicted Emotion: {emotion}")
```

## 📊 Visualizations

The notebook generates various visualizations including:

- 📊 **Class Distribution:** Bar charts showing emotion distribution
- 📈 **Performance Metrics:** Precision, Recall, F1-Score per emotion
- 🔥 **Confusion Matrices:** Heatmaps for each model
- 📉 **ROC Curves:** Multi-class ROC analysis with AUC scores
- ☁️ **Word Clouds:** Visualization of frequent words per emotion
- 📐 **Model Comparison:** Side-by-side performance comparison

## 🔧 Key Features

- ✅ Comprehensive text preprocessing pipeline
- ✅ Multiple model implementations and comparison
- ✅ Detailed evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ Cross-validation for robust performance estimation
- ✅ Extensive visualizations for insights
- ✅ Per-class performance analysis
- ✅ ROC-AUC analysis for multi-class classification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 👤 Author

**Suyash Koirala**
- GitHub: [@sk-koirala](https://github.com/sk-koirala)

## 🙏 Acknowledgments

- Dataset inspiration from emotion classification research
- Scikit-learn documentation and community
- NLTK for natural language processing tools

---


