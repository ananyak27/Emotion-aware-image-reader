# Emotion-Aware Image Reader Using Artificial Intelligence

An AI-powered web application that extracts text from images, identifies the underlying emotion of the extracted text using a fine-tuned DistilBERT model, and converts it into expressive speech using emotion-specific voices.

The application integrates Optical Character Recognition (OCR), Natural Language Processing (NLP), and Text-to-Speech (TTS) technologies to provide an intelligent and accessible reading experience.

---

## Features

- Extracts text from both printed and handwritten images.
- Automatically selects the most suitable OCR engine for improved recognition accuracy.
- Detects emotions from the extracted text using a fine-tuned DistilBERT model.
- Generates expressive speech using emotion-specific voices.
- Supports both image upload and reading images stored locally.
- User-friendly Flask-based web interface.
- Secure API key management using environment variables.

---

## Technology Stack

### Backend
- Python
- Flask

### Machine Learning & NLP
- DistilBERT
- Hugging Face Transformers
- PyTorch
- Scikit-learn

### OCR
- Tesseract OCR
- Microsoft TrOCR

### Text-to-Speech
- ElevenLabs API
- Murf AI API

### Other Libraries
- Pillow
- NLTK
- python-dotenv
- Requests

---

## Project Structure

```text
MINIFINALCODE/
│
├── emotion_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
│   └── training_args.bin
│
├── images/
├── sounds/
│
├── app.py
├── frontend.html
├── label_encoder.pkl
├── label_mapping.json
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How It Works

1. The user uploads an image or selects an image from the local images folder.
2. The application automatically determines the best OCR engine for the image.
3. Text is extracted from the image.
4. The extracted text is preprocessed.
5. A fine-tuned DistilBERT model predicts the emotion expressed in the text.
6. The detected emotion is mapped to an appropriate expressive voice.
7. Audio is generated using ElevenLabs or Murf AI.
8. The generated speech is returned to the user through the web interface.

---

## Supported Emotions

- Happiness
- Sadness
- Anger
- Fear
- Love
- Surprise
- Disgust
- Calm
- Hope
- Loneliness

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ananyak27/Emotion-aware-image-reader.git
cd Emotion-aware-image-reader
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` File

Create a `.env` file in the project root and add the required environment variables.

Example:

```env
ELEVENLABS_API_KEY=YOUR_ELEVENLABS_API_KEY
MURF_API_KEY=YOUR_MURF_API_KEY
TESSERACT_CMD="Add the path of the tesseract installed"
```



## Running the Application

Start the Flask server by running:

```bash
python app.py
```

The application will automatically open in your default web browser.

---

## Model Information

This project uses a fine-tuned DistilBERT model for emotion classification.

The model is stored in the `emotion_model` directory.


## Future Enhancements

- Support for additional languages
- Real-time camera-based image reading
- Improved OCR accuracy for complex documents
- Additional emotion categories
- Mobile application support

---

## Contributers

- Ananya Hebbar
- Abhijna V N 
- P Harshitha
