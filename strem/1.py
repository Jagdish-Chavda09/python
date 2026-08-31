import streamlit as st
import pandas as pd

st.title("Learning the streamlit lib")
st.header("Reading files from pandas")
name = st.text_input("Enter your name")
st.write("Hello ", name)


