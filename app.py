import streamlit as st
from operator import index
import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling
from PIL import Image
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="AutoBot",
    page_icon="🤖",
    layout="wide",

)

# Nav Bar
selected = option_menu(
    menu_title="AutoBot",
    options=["Dashboard", "Classification", "Regression", "Sample Applications"],
    icons=["boxes", "layout-wtf", "graph-up-arrow", "grid-fill"],
    menu_icon="stack",
    orientation="horizontal")

if selected == "Dashboard":
    body = """
    <h1> <marquee behavior="alternate">Hi! there👋 I am Autobot</marquee></h1>
    <p>Embark on your ML journey with me, and don't you worry you dont need to have any superpowers.
    I will be there helping you out just bring the data to me I am your Sherlock😉</p>
    <h3>Who am I, you ask❓</h3>
    <p>I am Autobot, an innovative AutoML web application designed to make machine learning accessible to everyone. With
     me, you can train machine learning models without writing a single line of complex code. I am user-friendly and 
     intuitive, making it easy for both beginners and experts to create, train, and deploy models. Whether you’re 
     looking to predict sales, find out more about a dataset, or anything in between,I’m here to help you achieve your goals with
      the power of machine learning. Let’s start this exciting journey together!</p>
    <h3>How can I make your life easier❓</h3>
    <p>I simplify your life by enabling on-the-go dataset analysis and model training without complex coding. I save 
    you time and effort by handling the technical aspects of machine learning, so you can focus on interpreting results 
    and making data-driven decisions. I make machine learning accessible to everyone, regardless of their background in
     data science or programming. By democratizing machine learning, I aim to make it beneficial for all. Let’s start 
     this exciting journey together!</p>
    """
    la1, la2 = st.columns(2)
    with la1:
        st.image("assets/autobot.gif")
    with la2:
        st.markdown(body, unsafe_allow_html=True)
    st.markdown("<hr style='border:1px dashed #FF4B4B'>", unsafe_allow_html=True)

# Classification Trainer Code

if selected == "Classification":

    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    with st.sidebar:
        st.image("assets/Core.gif")
        st.title("AutoBot : Classification Trainer")
        choice = st.radio(
            "Workflow 👇", ["Upload", "Profiling", "Modelling", "Download"])
        st.info("This is an AutoML app for Classification problems just upload a dataset and go through the selection "
                "steps only this time let our app do all the hardworking.")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        st_profile_report(profile_df)

    if choice == "Modelling":
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

    if choice == "Download":
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")

# Regression Trainer Code

if selected == "Regression":

    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    with st.sidebar:
        st.image("assets/Core.gif", use_column_width="always")
        st.title("AutoBot : Regression Trainer")
        choice = st.radio(
            "Workflow 👇", ["Upload", "Profiling", "Modelling", "Download"])
        st.info("This is an AutoML app for Regression problems just upload a dataset and go"
                " through the selection steps only this time let our app do all the hardworking.")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        st_profile_report(profile_df)

    if choice == "Modelling":
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

    if choice == "Download":
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")

if selected == "Sample Applications":
    options = option_menu(
        menu_title=None,
        options=["About", "Employee Churn Analysis", "Bitcoin Price Prediction", "Player Price Prediction", "Announcements"],
        icons=["body-text", "briefcase", "coin", "dribbble", "megaphone"], orientation="horizontal")
    # ANNOUNCEMENTS
    if options == "Announcements":
        col1, col2, col3 = st.columns([2, 6, 2])

        with col1:
            st.write("")

        with col2:
            st.markdown("""<br>""", unsafe_allow_html=True)
            st.image("assets/updates.gif")
            st.markdown("""<h1>More Updates On The Way . . . . """, unsafe_allow_html=True)
