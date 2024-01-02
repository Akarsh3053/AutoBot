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
    aa1, aa2 = st.columns(2)
    with aa1:
        body = """
            <h1>Here is how you can gain my superpowers✨</h1>
            <p>Together we can train a machine learning model in a few simple steps:</p>
            <h3>Upload Your Dataset📑:</h3>
            <p>Begin by uploading your dataset using the "Upload" section. Click on the "Upload Your Dataset" button and
             select your dataset <b>CSV<b> file.</p>
            <h3>I will generate a detailed EDA📋:</h3>
            <p>Once your dataset is uploaded, explore its characteristics through the "Profiling" section.
            Autobot performs EDA on the dataset so that you know more about it.</p>
            <h3>Train Your Model🧮:</h3>
            <p>Move on to the "Modelling" section and choose the target column then from here Autobot does all the work 
            it trains, compares different models, and selects the best-performing one.</p>
            <h3>Download Your Model💾:</h3>
            <p>After all this, a best performing trained model is ready, download it and use wherever you want.</p> 
            """
        st.markdown(body, unsafe_allow_html=True)
    with aa2:
        ins = Image.open("assets/manual.png")
        st.image(ins.resize((700, 700)))

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
        options=["About", "Employee Churn Analysis", "Bitcoin Price Prediction", "Announcements"],
        icons=["body-text", "briefcase", "coin", "megaphone"], default_index=3,orientation="horizontal")
    # About
    if options == "About":
        la1, la2 = st.columns(2)
        with la1:
            st.image("assets/core.gif")
        with la2:
            h1 = """
                    <h1 style="padding-top:65px">
                    Sample Model Significance
                    </h1>
                    <p>I will also provides some sample models just to showcase how machine learning can really enable 
                    you to see into the future by making 
                    predictions on data by learning from it. These are just to showcase for displaying the capabilities
                    the models trained here.
                    </p>
                    """
            st.markdown(h1, unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)

    # ANNOUNCEMENTS
    if options == "Announcements":
        col1, col2, col3 = st.columns([2, 6, 2])

        with col1:
            st.write("")

        with col2:
            st.markdown("""<br>""", unsafe_allow_html=True)
            st.image("assets/updates.gif")
            st.markdown("""<h1>More Updates On The Way . . . . """, unsafe_allow_html=True)
