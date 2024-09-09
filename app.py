import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Attempt to import missingno and handle the ImportError
try:
    import missingno as msno
    missingno_installed = True
except ImportError:
    missingno_installed = False

from sklearn.cluster import KMeans

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("Mall_Customers.csv")
    return data

data = load_data()

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f4f4f4;
        color: #333;
    }
    .streamlit-expanderHeader {
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        color: #333;
    }
    .sidebar .sidebar-content {
        background-color: #dcdcdc;
    }
    .stButton button {
        background-color: #ff5722;
        color: white;
        font-weight: bold;
    }
    .stTitle {
        font-family: 'Arial', sans-serif;
        color: #222;
    }
    .stSubheader {
        font-family: 'Arial', sans-serif;
        color: #555;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Customer Segmentation Analysis")

# Sidebar with project description and acknowledgment
st.sidebar.header("Project Description")
st.sidebar.write("""
Customer segmentation is the process of uncovering insights about a firm's customer base based on their interactions and purchase behavior. This analysis aims to identify target customers to help the marketing team tailor their strategies effectively.

In this project, we explore various data visualizations and perform K-Means clustering to segment customers based on their age, annual income, and spending score.
""")
st.sidebar.write("Model by Engineer Shahan Nafees")

st.sidebar.header("Filters")
selected_option = st.sidebar.selectbox("Select Visualization", 
                                          ["Data Overview", "Data Visualization", "K-Means Clustering"])

if selected_option == "Data Overview":
    st.subheader("Data Overview")
    st.write("### Data Head")
    st.write(data.head())
    st.write("### Data Shape")
    st.write(data.shape)
    st.write("### Data Description")
    st.write(data.describe())
    st.write("### Data Info")
    st.write(data.info())
    st.write("### Missing Values")
    st.write(data.isnull().sum())
    if missingno_installed:
        st.write("### Missing Data Visualization")
        fig, ax = plt.subplots()
        msno.matrix(data, ax=ax)
        st.pyplot(fig)
    else:
        st.write("Missingno is not installed. Please install it to see the missing data visualization.")

elif selected_option == "Data Visualization":
    st.subheader("Data Visualization")
    
    st.write("### Distribution of Age")
    fig, ax = plt.subplots()
    sns.histplot(data['Age'], kde=True, ax=ax)
    ax.set_title("Distribution of Age")
    st.pyplot(fig)
    
    st.write("### Distribution of Annual Income")
    fig, ax = plt.subplots()
    sns.histplot(data['Annual Income (k$)'], kde=True, ax=ax)
    ax.set_title("Distribution of Annual Income")
    st.pyplot(fig)
    
    st.write("### Distribution of Spending Score")
    fig, ax = plt.subplots()
    sns.histplot(data['Spending Score (1-100)'], kde=True, ax=ax)
    ax.set_title("Distribution of Spending Score")
    st.pyplot(fig)
    
    st.write("### Gender Distribution")
    fig, ax = plt.subplots()
    df = data.groupby('Gender').size()
    df.plot(kind='pie', autopct='%.2f%%', colors=['lightgreen', 'orange'], ax=ax)
    ax.set_title("Gender Distribution")
    st.pyplot(fig)
    
    st.write("### Pairplot")
    fig = sns.pairplot(data, hue='Gender').fig
    st.pyplot(fig)

elif selected_option == "K-Means Clustering":
    st.subheader("K-Means Clustering")
    
    st.write("### Elbow Method for Optimal Clusters")
    x = data.iloc[:, [3, 4]].values
    k = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(x)
        k.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), k)
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    st.write("### K-Means Clustering Result")
    num_clusters = st.slider("Select number of clusters", 1, 10, 5)
    model = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
    y_kmeans = model.fit_predict(x)

    fig, ax = plt.subplots()
    for i in range(num_clusters):
        ax.scatter(x[y_kmeans == i, 0], x[y_kmeans == i, 1], s=100, label=f'Cluster {i + 1}')
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='black', label='Centroids')
    ax.set_title("K Means Clustering")
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.legend()
    st.pyplot(fig)
