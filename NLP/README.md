# 📰 Low-Resource Fake News Detection using LSTM

This project focuses on detecting fake news in **low-resource languages** using a simple LSTM-based deep learning model. Designed and trained on the `banFakeNews` dataset (Bengali), this model demonstrates the power of neural networks even when limited linguistic resources are available.

## 🔍 Project Overview

Low-resource languages often lack large labeled datasets, making it challenging to apply standard NLP methods. This project tackles the fake news classification problem by:

- Leveraging LSTM networks for text sequence modeling
- Using basic embeddings without transformers
- Building a web interface for easy interaction and testing

---

## 🧠 Model Architecture

```python
class FakeNewsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FakeNewsLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        final_out = self.fc(lstm_out[:, -1, :])
        return final_out
# 🧠 Fake News Detection in Bengali using LSTM

## 📌 Model Details

- **Input:** Word embeddings (pretrained or learned)  
- **Hidden Layer:** LSTM  
- **Output:** Binary label (Real or Fake)  

## 📂 Dataset

**banFakeNews Dataset** – A Bengali fake news classification dataset.

### Contains:
- News headlines  
- Labels: `0` (Real), `1` (Fake)  
- Preprocessing steps include:
  - Tokenization  
  - Padding  
  - Word embedding  

## 📊 Results

| Metric     | Score |
|------------|-------|
| Accuracy   | XX%   |
| F1 Score   | XX%   |
| Precision  | XX%   |
| Recall     | XX%   |

*Fill in your real metrics after model training.*

## 🚀 Running the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/low_resource_nlp.git
   cd low_resource_nlp
2. Install Dependencies
   pip install -r requirements.txt
3. Train the Model
   python train.py
4. Launch Web App (Streamlit)
   streamlit run app.py**
**📓 Kaggle Notebook (Model + Training):**
    https://www.kaggle.com/code/jobinsabu/notebook955af55450
**💻 Streamlit App (Local testing):**
   streamlit run app.py

**💡 Future Plans**
   Train with transformers (e.g., IndicBERT, mBERT)
   Augment dataset with translated or synthetic examples
   Try transfer learning or fine-tuning approaches
   Deploy fully on Hugging Face with GPU support

**Tools & Frameworks**
  PyTorch
  Streamlit
  Kaggle
  Python NLP Libraries (NLTK, Scikit-learn, etc.)

🙋 Author
👤 Jobin-Sabu

Pre Final-year undergraduate with deep interest in NLP & AI research
Actively experimenting with low-resource language problems and democratizing AI access

📜 License
Apache License 2.0

Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
If you'd like to collaborate, contribute, or provide feedback, feel free to open issues or reach out on GitHub!
