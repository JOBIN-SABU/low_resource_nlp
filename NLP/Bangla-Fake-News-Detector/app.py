import streamlit as st
import torch
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
from nltk.corpus import stopwords


nltk.download("punkt")
nltk.download("stopwords")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "models/fake_news_model.pth"
vectorizer_path = "models/tfidf_vectorizer.pkl"


class FakeNewsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FakeNewsLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        final_out = self.fc(lstm_out[:, -1, :])
        return final_out


model = FakeNewsLSTM(input_size=1000, hidden_size=128, output_size=2)  
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Bangla stopwords for better preprocessing
bangla_stopwords = set(stopwords.words('bengali'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text) 
    tokens = nltk.word_tokenize(text)  
    tokens = [word for word in tokens if word not in bangla_stopwords]  # Remove stopwords
    return " ".join(tokens)


st.title("Bangla Fake News Detection 🚀")
st.write("Paste a Bangla news article below and check if it's **Real or Fake**!")


user_input = st.text_area("Enter News Article Here", "")


if st.button("Check News"):
    if user_input.strip():  
        processed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([processed_text]).toarray()
        input_tensor = torch.tensor(vectorized_text, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        result = " Real News" if prediction == 1 else "Fake News"
        st.success(result)
    else:
        st.warning(" Please enter a valid news article!")
