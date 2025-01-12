# BPO Customer Care Models

This repository contains five different models for sentiment and emotion analysis, including a GAN model for generating synthetic customer service data. These models can be used for various BPO (Business Process Outsourcing) applications, such as customer care, sentiment analysis, and more.

## Table of Contents
1. [Models](#models)
2. [Usage](#usage)
3. [Requirements](#requirements)
4. [Model Details](#model-details)
5. [Future Improvements](#future-improvements)
6. [Citations](#citations)
7. [License](#license)
8. [Contact](#contact)

---

## Models
1. **Tamil Slang Normalization Model** (`Rajkumar57/TamilSlangNormalization`)
2. **Tamil Emotion Analysis Model** (`Rajkumar57/tamilsentiment-model`)
3. **English Emotion Analysis Model** (`Rajkumar57/englishsentiment-model`)
4. **GAN + CNN Model for Tamil Audio Emotion Prediction** (Custom)
5. **GAN + CNN Model for English Audio Emotion Prediction** (Custom)

---

## Usage

### 1. Tamil Slang Normalization Model
This model normalizes Tamil slang and informal text to standard Tamil, because in different places Tamil words are pronounced differently. Trained and uploaded on Hugging Face.

#### Example:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_dir = "Rajkumar57/TamilSlangNormalization"  
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def normalize_text(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=32, truncation=True).to(model.device)
    outputs = model.generate(inputs["input_ids"])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
slang_text = "சரி செம்ம"
normalized = normalize_text(slang_text, model, tokenizer)
print(f"Input: {slang_text} -> Normalized: {normalized}")
```

### 2. Tamil Sentiment Analysis Model
This model analyzes emotion (like sadness, joy, love, anger, fear, and surprise) in Tamil text.

#### Example:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib

model_name = "Rajkumar57/tamilsentiment-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "இந்த ஒரு உதாரணம்."

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class_idx = torch.argmax(logits, dim=-1).item()

# Note: You'll need to load your label encoder or map the index to appropriate sentiment labels
```

### 3. English Emotion Analysis Model
This model analyzes emotion (like sadness, joy, love, anger, fear, and surprise) in English text.

#### Example:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained("Rajkumar57/englishsentiment-model")
model = AutoModelForSequenceClassification.from_pretrained("Rajkumar57/englishsentiment-model")

def predict_emotion(texts):
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, axis=1).numpy()
    return [emotion_labels[pred] for pred in predictions]

# Example usage
texts = [
    "I feel really happy today!",
    "I'm so angry right now!",
    "This is the worst day ever."
]
predictions = predict_emotion(texts)
for text, emotion in zip(texts, predictions):
    print(f"Text: {text} => Predicted Emotion: {emotion}")
```

### 4. GAN + CNN Model for Tamil Audio Emotion Prediction
This model uses a GAN architecture to augment training data and a CNN model to predict emotions from audio features extracted using Librosa.

#### Example:
```python
import librosa
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder

# Load and preprocess audio
filename = 'example_audio.wav'
y, sr = librosa.load(filename, sr=None)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
input_features = np.mean(mfccs.T, axis=0).reshape(1, -1)

# Define a simple CNN model for emotion prediction
class AudioEmotionCNN(nn.Module):
    def __init__(self):
        super(AudioEmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(40 * 16, 64)
        self.fc2 = nn.Linear(64, 6)  # Assuming 6 emotion classes

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate and use the model
emotion_model = AudioEmotionCNN()
audio_tensor = torch.tensor(input_features, dtype=torch.float32)
output = emotion_model(audio_tensor)

detected_emotion = torch.argmax(output, dim=1).item()
label_encoder = LabelEncoder()
label_encoder.fit(['happy', 'sad', 'angry', 'fearful', 'surprised', 'neutral'])
print(f"Predicted Emotion: {label_encoder.inverse_transform([detected_emotion])[0]}")
```

---

### 5. GAN + CNN Model for English Audio Emotion Prediction
This model uses a GAN architecture to augment training data and a CNN model to predict emotions from audio features extracted using Librosa.

#### Example:
```python
import librosa
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder

# Load and preprocess audio
filename = 'example_audio.wav'
y, sr = librosa.load(filename, sr=None)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
input_features = np.mean(mfccs.T, axis=0).reshape(1, -1)

# Define a simple CNN model for emotion prediction
class AudioEmotionCNN(nn.Module):
    def __init__(self):
        super(AudioEmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(40 * 16, 64)
        self.fc2 = nn.Linear(64, 6)  # Assuming 6 emotion classes

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate and use the model
emotion_model = AudioEmotionCNN()
audio_tensor = torch.tensor(input_features, dtype=torch.float32)
output = emotion_model(audio_tensor)

detected_emotion = torch.argmax(output, dim=1).item()
label_encoder = LabelEncoder()
label_encoder.fit(['happy', 'sad', 'angry', 'fearful', 'surprised', 'neutral'])
print(f"Predicted Emotion: {label_encoder.inverse_transform([detected_emotion])[0]}")
```

## Requirements

- `transformers`
- `torch`
- `numpy`
- `librosa`
- `joblib` (for Tamil sentiment model if using label encoder)

To install the dependencies:

```bash
pip install transformers torch numpy librosa joblib
```

---

## Model Details

### 1. Tamil Slang Normalization
- Based on mBART-large-50 and LSTM
- Trained on Tamil slang dataset
- Max sequence length: 32 tokens

### 2. Tamil Sentiment Analysis
- Based on XLM-RoBERTa, LSTM, and CNN
- Trained on Tamil sentiment dataset
- Supports multiple sentiment categories

### 3. English Emotion Analysis
- Based on XLM-RoBERTa, LSTM, and CNN
- Trained on emotion-labeled English text
- Supports 6 emotion categories

### 4. GAN + CNN Tamil Audio Emotion Model
- GAN used for data augmentation
- CNN used for audio feature-based emotion classification

### 5. GAN + CNN English Audio Emotion Model
- GAN used for data augmentation
- CNN used for audio feature-based emotion classification

---

## Future Improvements
- Develop a slang normalization model for English.
- Add support for additional languages like Hindi and Kannada.
- Fine-tune LLaMA 3.1 for extracting detailed information from customer service text data.

---

## Citations

If you use these models in your research, please cite:

```
@misc{rajkumar2024bpo,
  author = {Rajkumar, G P},
  title = {BPO Customer Care Models},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/Rajkumar57}}
}
```

---

## License

These models are released under the MIT License.

---

## Contact

For issues and questions, please open an issue on the GitHub repository.
