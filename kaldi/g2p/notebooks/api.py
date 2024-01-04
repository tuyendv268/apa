import streamlit as st
import pandas as pd
from arpa_to_ipa import arpa_to_ipa
st.title('PREP IPA CONVERTER')

path = "/data/codes/apa/kaldi/stt/resources/lexicon"
lexicon = pd.read_csv(path, names=["word", "arpa"], sep="\t")
lexicon["ipa"] = lexicon.arpa.apply(arpa_to_ipa)

vocab = set(lexicon.word.tolist())

text = st.text_input("Nhập văn bản")

if st.button("Submit"):
    word = text.strip().upper()

    if word not in vocab:
        st.write("out of vocab")
    else:
        output = lexicon[lexicon.word == word]
        st.write("output: ")
        st.write(output)
        st.write(output.to_dict())