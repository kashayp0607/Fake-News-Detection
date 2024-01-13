import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np
import plotly.express as px

MODEL_PATH = r"LSTM_model_1.h5"
MAX_SEQUENCE_LENGTH = 200
tokenizer_file = "tokenizer.pickle"

wordnet = WordNetLemmatizer()
regex = re.compile('[%s]' % re.escape(string.punctuation))

with open(tokenizer_file, 'rb') as handle:
    tokenizer = pickle.load(handle)


@st.cache_data()
def Load_model():
    model = load_model(MODEL_PATH)
    model.summary()  # included to make it visible when the model is reloaded
    return model




if __name__ == '__main__':
    st.title('Fake News Classification app ')
    st.write("A simple fake news classification app utilising an LSTM model")
    st.info("LSTM model and tokeniser loaded ")
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "Some news", height=200)
    predict_btt = st.button("predict")
    model = Load_model()


    def basic_text_cleaning(line_from_column):
        # This function takes in a string, not a list or an array for the arg line_from_column

        tokenized_doc = word_tokenize(line_from_column)

        new_review = []
        for token in tokenized_doc:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)

        new_term_vector = []
        for word in new_review:
            if not word in stopwords.words('english'):
                new_term_vector.append(word)

        final_doc = []
        for word in new_term_vector:
            final_doc.append(wordnet.lemmatize(word))

        return ' '.join(final_doc)


    if predict_btt:
        clean_text = []
        model = Load_model()
        i = basic_text_cleaning(sentence)
        clean_text.append(i)
        sequences = tokenizer.texts_to_sequences(clean_text)
        data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
        prediction = model.predict(data)
        # Rest of your prediction code...

        prediction_class = np.argmax(prediction, axis=-1)[0]

        st.header("Prediction using LSTM model")

        if prediction_class == 0:
            st.success('This is not a fake news')
        else:
            st.warning('This is a fake news')

        class_labels = ["fake", "true"]
        prob_list = [prediction[0][1] * 100, prediction[0][0] * 100]
        prob_dict = {"true/fake": class_labels, "Probability": prob_list}
        df_prob = pd.DataFrame(prob_dict)
        fig = px.bar(df_prob, x='true/fake', y='Probability')
        model_option = "LSTM"
        st.plotly_chart(fig, use_container_width=True)
