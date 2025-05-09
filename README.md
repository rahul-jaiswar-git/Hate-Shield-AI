# Multi-Modal Hate Speech Detection in Hinglish

![GitHub repo size](https://img.shields.io/github/repo-size/rahul-jaiswar-git/Hate-Shield-AI?style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/rahul-jaiswar-git/Hate-Shield-AI?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/rahul-jaiswar-git/Hate-Shield-AI?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/rahul-jaiswar-git/Hate-Shield-AI?style=for-the-badge)
![GitHub pull requests](https://img.shields.io/github/issues-pr/rahul-jaiswar-git/Hate-Shield-AI?style=for-the-badge)

![Project Demo](Frontend/models.gif)

> A multi-modal hate speech detection system for Hinglish (Hindi-English code-mixed language). This Streamlit-based application allows users to detect hate speech in text, audio, video, and images using deep learning, NLP, OCR, and speech-to-text.

## Table of Contents
- [What is Hate Speech Detection?](#what-is-hate-speech-detection)
- [Key Features](#key-features)
- [💻 Prerequisites](#-prerequisites)
- [🚀 Installation](#-installation)
- [⬇️ Download the Model](#️-download-the-model)
- [☕ Usage](#-usage)
- [🛠️ Technology Stack](#️-technology-stack)
- [🏗️ File Structure](#️-file-structure)
- [🤝 Contributors](#-contributors)
- [📝 License](#-license)

---

## What is Hate Speech Detection?

Hate speech detection is the process of identifying and classifying content (text, audio, video, images, etc.) as hate speech or non-hate speech. This project focuses on Hinglish, a code-mixed language, and supports detection across multiple modalities using advanced machine learning and NLP techniques.

### Key Features

- [x] **Multi-Modal Detection:** Supports text, audio, video, and image hate speech detection in Hinglish.
- [x] **Real-Time Inference:** Fast, interactive predictions via a modern Streamlit web interface.
- [x] **Robust Preprocessing:** Advanced text cleaning, OCR for images, and speech-to-text for audio/video.
- [x] **Transfer Learning:** Utilizes a fine-tuned BERT model for high-accuracy classification.
- [x] **User-Friendly UI:** Intuitive navigation and clear results for all input types.
- [x] **Modular Codebase:** Easily extendable for new modalities or languages.

## 💻 Prerequisites

- Python 3.8+
- pip

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahul-jaiswar-git/Hate-Shield-AI.git
   cd Hate-Shield-AI
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the model files:**
   - Download all model files from the Hugging Face repository:
     [Hinglish-based-Hate-Speech-detection-model-v1 on Hugging Face](https://huggingface.co/rahuljaiswarofficial/Hinglish-based-Hate-Speech-detection-model-v1)
   - Place all downloaded files (e.g., `tf_model.h5`, `config.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`, `label_encoder.pkl`) into the `hate_speech_model/` directory.

## ⬇️ Download the Model

You can download the pre-trained model and all required files from Hugging Face:

[https://huggingface.co/rahuljaiswarofficial/Hinglish-based-Hate-Speech-detection-model-v1](https://huggingface.co/rahuljaiswarofficial/Hinglish-based-Hate-Speech-detection-model-v1)

**After downloading, place all files in the `hate_speech_model/` directory before running the app.**

## ☕ Usage

Run the Streamlit app:
```bash
streamlit run app.py
```
- Use the sidebar to navigate between Home, Model Check, and About Us.
- In "Check Model", select the desired classification task (Text, Audio, Video, Image).
- Upload files or enter text as prompted.

## 🛠️ Technology Stack

- **User Interface:**
  - Streamlit (web app framework)
  - Pillow (image handling)
- **Machine Learning & NLP:**
  - TensorFlow (deep learning backend)
  - HuggingFace Transformers (BERT model)
  - joblib (model serialization)
  - numpy (numerical operations)
- **Audio & Speech Processing:**
  - SpeechRecognition (speech-to-text)
  - pydub, librosa, soundfile (audio file handling)
- **Image & Video Processing:**
  - OpenCV (image/video processing)
  - pytesseract (OCR)
- **Utilities:**
  - requests (API calls)

## 🏗️ File Structure

```
project/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── styles.css              # Custom styles for Streamlit
├── hate_speech_model/      # Model files and label encoder
├── Classifier/             # Classification modules for each modality
│   ├── text_classification.py
│   ├── audio_classification.py
│   ├── video_classification.py
│   ├── image_classification.py
├── Frontend/               # Images and GIFs for UI
│   ├── models.gif
│   ├── about us.gif
│   └── Hate.jpg
├── Dataset/                # (Optional) Data for training/testing
├── Eg Data/                # (Optional) Example data for demo
└── ...
```

## 🤝 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/rahul-jaiswar-git" title="Rahul Jaiswar">
        <img src="https://avatars.githubusercontent.com/rahul-jaiswar-git" width="100px;" alt="Rahul Jaiswar's GitHub photo"/><br>
        <sub>
          <b>Rahul Jaiswar</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

**Special Thanks:** Open-source community, HuggingFace, and all referenced libraries

## 📝 License

This project is licensed under the [MIT License](https://github.com/rahul-jaiswar-git/Hate-Shield-AI/blob/main/LICENSE). 