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
        <h1 >Say Hi to the AutoBot</h1>
        <p>It is a web-based AutoML application. This application is designed to train Machine 
        Learning models on a provided dataset. The images you see represent various aspects of the project. The crystal ball 
        image symbolizes the predictive power of machine learning models, which can "see" patterns in data and make 
        predictions about future data. The high-tech theme of the images reflects the advanced algorithms and computational 
        processes involved in training these models. The web-based nature of the application means that it can be accessed 
        from anywhere with an internet connection, making it highly accessible and user-friendly. Users can upload their 
        datasets, select the type of model they want to train, and then let the application handle the rest. The application 
        will automatically preprocess the data, select the best model parameters, and train the model. Once the model is 
        trained, users can download it for use in their own projects or use it directly within the application to make 
        predictions on new data. Overall, AutoBot represents a powerful tool for anyone looking to 
        leverage the power of machine learning, whether they are experienced data scientists or beginners just starting out 
        in the field.</p>
        """
    la1, la2 = st.columns(2)
    with la1:
        st.image("assets/autobot.gif")
    with la2:
        st.markdown(body, unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
    aa1, aa2 = st.columns(2)
    with aa1:
        body = """
            <h1>Instructions to Use</h1>
            <p>Using AutoBot is a straightforward process that involves a few key steps:</p>
            <h3>Upload Your Dataset:</h3>
            <p>Begin by uploading your dataset using the "Upload" section. Click on the "Upload Your Dataset" button and select your dataset file. 
            Crystal Ball supports various file formats, including CSV.</p>
            <h3>Get Your EDA</h3>
            <p>Once your dataset is uploaded, explore its characteristics through the "Profiling" section.
            Crystal Ball performs Exploratory Data Analysis (EDA) to provide insights into the dataset's structure, statistics, and patterns.</p>
            <h3>Train Your Model</h3>
            <p>Move on to the "Modelling" section to initiate the model training process. 
            Choose the target column for your classification or regression task.
            Crystal Ball automates the setup, compares different models, and selects the best-performing one.</p>
            <h3>Download Your Model:</h3>
            <p>After the model is trained, download it for use in your projects. Utilize the "Download" section to obtain the trained model file.</p> 
            """
        st.markdown(body, unsafe_allow_html=True)
    with aa2:
        ins = Image.open("images/instructions.jpg")
        st.image(ins.resize((700, 700)))
    st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
    st.markdown("""<h1 style="text-align:center">How Crystal Ball Works</h1>""", unsafe_allow_html=True)
    st.markdown("""<p>The provided code uses the PyCaret library for AutoML tasks in both classification and regression scenarios.
         PyCaret internally utilizes a variety of machine learning algorithms, such as Decision Trees, 
         Random Forest, Gradient Boosting, Support Vector Machines, K-Nearest Neighbors, and more. 
         The library automatically selects and tunes algorithms based on dataset characteristics and user preferences.</p>
         <p>It's worth noting that users 
         can customize the selection of
         algorithms and other parameters according to their specific requirements and preferences.</p>
         <p>The AutoML pipeline in the project is implemented by PyCaret. 
         It is an open-source, low-code machine-learning library in Python that automates machine-learning workflows.
          It is designed to make performing standard tasks in a machine learning project easy1. 
          PyCaret is a Python version of the Caret machine learning package in R. 
         It is popular because it allows models to be evaluated, compared, and tuned on a given dataset with just a 
         few lines of code.</p>""", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
    st.markdown("""<h1>Explore More</h1>""", unsafe_allow_html=True)
    bb1, bb2 = st.columns(2)
    with bb1:
        body = """
            <h3>Explore Sample Applications:</h3>
            <p>If you're new to machine learning or want to explore predefined use cases, check out the "Sample Applications" section. Crystal Ball provides sample 
            applications like <ul><li> Player Price Prediction</li> <li>Bitcoin Price Prediction</li> and more. 
            Simply follow the prompts and make predictions based on the provided models.</p>
            """
        st.markdown(body, unsafe_allow_html=True)
        body2 = """
                    <h3>Navigate Between Sections:</h3>
                    <p>Use the navigation bar on the left to switch between different sections <ul> <li>Dashboard</li> <li>Classification</li> <li>Regression</li>  <li>Sample Applications</li></ul>
                    Now you're ready to unleash the power of Crystal Ball for your machine learning tasks.
                     Whether you're a data science expert or just starting, Crystal Ball makes the process intuitive and efficient.</p>
                    """
        st.markdown(body2, unsafe_allow_html=True)
    with bb2:
        logo = Image.open("images/mixoo.jpg")
        st.image(logo.resize((650, 500)))

    st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
    st.markdown("""<h1>Features of Crystal Ball :</h1>""", unsafe_allow_html=True)
    zz1, zz2 = st.columns(2)
    with zz1:
        body = """ 
            <ul style="list-style-type:none">
            <li><h3>Interactive Visualizations:</h3>Explore data patterns and model insights through interactive visualizations using advanced libraries like Plotly and seaborn. 
            Users can now interactively analyze predictions and gain a deeper understanding of their datasets with visuals like bargraphs, heatmaps and correlation matrices.</li>
            <br>
            <li><h3>Web Based ML Dashboard:</h3>Crystal ball is deployed as a web-app which can be accessed on any device,
             allowing users to train machine learning models and generate reporst on datsets on the go. This will greately improve its accessibility
              as well as it will make ML avaialble to all.</li>
             <br>
             <li><h3>Model Deployment:</h3>Final output of crystal ball is a trained models,so that you can deploy these models into any machine learning application easily. 
              These pretrained pickled models a can be easily integrated into language independent machine learning apps to be deployed in projects of all scales.</li>
            </ul>
            """
        st.markdown(body, unsafe_allow_html=True)
    with zz2:
        body = """
            <ul style="list-style-type:none">
            <li><h3>Automated EDA:</h3>The framework performs exploratory data analysis on all the input datasets to 
            generate a comprehensive report on data parameters, outlyers, null values and high-correlations between the 
            data items so that user has 
            understanding of dataset and based on it can decide to whether move forward to train models or not.</li>
            <br>
             <li><h3>Documentation and Tutorials:</h3>Navigate Crystal Ball with ease. The comprehensive documentation and tutorials guide users through the application's features, machine learning concepts, 
            and provide practical examples for enhanced user understanding.</li>
            <br>
             <li><h3>Model Statistics:</h3>The dashboard provides all performance stats for all trained models to correctly
              back the claims for best models which it provides in download section enabling the users to understand, 
              how a particular model is better performing than rest. </li>
            """
        st.markdown(body, unsafe_allow_html=True)
    st.markdown("<hr style='border:1px dashed black'>", unsafe_allow_html=True)
    st.markdown(
        """<br><b style=" margin-bottom:0px padding-bottom:0px">Crystall Ball aims to make machine learning accessible to all as an Intutive user friendly tool which lets everyone use powers of Machine Learning irrespective of their coding and technical skills.</b>""",
        unsafe_allow_html=True)

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
    pass
