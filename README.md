# 🖼️ Image Caption Generator (Flickr8k)

An end-to-end **Image Caption Generator** project built using **Deep Learning (PyTorch)**.  
The system generates natural language captions for images and provides a simple **web-based UI** for demonstration.

---

## 📌 Project Overview

This project implements a complete machine learning pipeline:
- Dataset preparation (Flickr8k)
- Text preprocessing and vocabulary creation
- LSTM-based caption generation model
- Model training and evaluation
- FastAPI backend for inference
- HTML + JavaScript frontend for user interaction

⚠️ **Note:**  
The current implementation is a **baseline caption generator**.  
The UI supports image upload, but captions are generated using learned language patterns (CNN-based image feature extraction can be added as a future enhancement).

---

## 🧠 Architecture

```
Image Upload (UI)
        ↓
 FastAPI Backend
        ↓
 LSTM Caption Generator
        ↓
 Generated Caption
```

---

## 📂 Project Structure

```
Image_Caption_Generator_Flickr8k/
│
├── app.py                 # FastAPI backend
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
│
├── data/
│   ├── raw/               # Flickr8k images & captions
│   └── processed/         # Processed vocabulary
│
├── models/
│   └── decoder.pth        # Trained model
│
├── src/
│   ├── preprocessing.py  # Text preprocessing & vocab
│   ├── dataset.py        # Custom PyTorch dataset
│   ├── model.py          # LSTM model
│   ├── train.py          # Training script
│   ├── inference.py      # Caption generation
│   ├── evaluate.py       # BLEU score evaluation
│   └── utils.py          # Helper functions
│
└── ui/
    ├── index.html         # Frontend UI
    └── app.js             # Frontend logic
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Image_Caption_Generator_Flickr8k.git
cd Image_Caption_Generator_Flickr8k
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Preprocess Data
```bash
python -m src.preprocessing
```

### Step 2: Train the Model
```bash
python -m src.train
```

### Step 3: Start Backend (FastAPI)
```bash
python -m uvicorn app:app --reload
```

Open API docs:
```
http://127.0.0.1:8000/docs
```

### Step 4: Run Frontend UI
```bash
cd ui
python -m http.server 5500
```

Open in browser:
```
http://localhost:5500/index.html
```

---

## 📊 Evaluation

BLEU score evaluation:
```bash
python -m src.evaluate
```

---

## 🎓 Technologies Used

- Python
- PyTorch
- FastAPI
- HTML, JavaScript
- NLTK
- Flickr8k Dataset

---

## 📌 Future Enhancements

- CNN + LSTM (ResNet-based image features)
- Improved caption quality
- Streamlit UI
- Deployment on cloud

---

## 👩‍💻 Author

**Bhagyashree Mali**  
AI & ML Engineering Student  

---

## ⭐ Acknowledgements

- Flickr8k Dataset
- PyTorch & FastAPI Documentation

---

## 📜 License

This project is for **academic and learning purposes**.
## 📧 Contact

**Bhagyashree Mali**

- 📧 Email: **BHAGYASHREEMALI1624@gmail.com**  
- 🔗 LinkedIn: [www.linkedin.com/in/bhagyashree-mali]  
- 🧠 GitHub: https://github.com/bhagyashreemali
- 📍 Location: Pune, Maharashtra, India

