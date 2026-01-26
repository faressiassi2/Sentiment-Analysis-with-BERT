import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re # regular expression
import string
import nltk
from nltk.util import pr
from nltk.corpus import stopwords

from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

data = pd.read_csv('Tweets.csv')
df = data[["text", "sentiment"]].dropna()
df["text"] = df["text"].astype(str)

stemmer = nltk.SnowballStemmer("english")
nltk.download('stopwords')
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    #text = [word for word in text.split(' ') if word not in stopword]
    #text = " ".join(text)
    #text = [stemmer.stem(word) for word in text.split(' ')]
    #text = " ".join(text)
    return text
  
df["text"] = df["text"].apply(clean)

# Encoder les labels
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
df["label"] = df["sentiment"].map(label_map)

# Separation train / test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
train_df.to_csv("df_train.csv", index=False)
test_df.to_csv("df_test.csv", index=False)

print("preprocessing over")
