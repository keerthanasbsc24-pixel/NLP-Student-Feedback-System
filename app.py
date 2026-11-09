
import streamlit as st
import pandas as pd
import joblib
import re
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Student Feedback System",
    page_icon="🎓",
    layout="wide"
)

# Load saved models
category_model = joblib.load("category_model.pkl")
category_vectorizer = joblib.load("category_vectorizer.pkl")

sentiment_model = joblib.load("sentiment_model.pkl")
sentiment_vectorizer = joblib.load("sentiment_vectorizer.pkl")

stop_words = joblib.load("stop_words.pkl")

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()

    stop_words = {
        'the', 'is', 'are', 'was', 'were', 'a', 'an',
        'and', 'or', 'of', 'to', 'in', 'on', 'for',
        'this', 'that', 'it', 'with', 'very', 'has',
        'have', 'had', 'be', 'been', 'am', 'i', 'we'
    }

    return ' '.join(
        word for word in words if word not in stop_words
    )


# Prediction function
def predict_feedback(feedback):
    cleaned = preprocess_text(feedback)

    if not cleaned.strip():
        return "Unable to classify", "Unable to classify"

    category_input = category_vectorizer.transform([cleaned])
    sentiment_input = sentiment_vectorizer.transform([cleaned])

    category = category_model.predict(category_input)[0]
    sentiment = sentiment_model.predict(sentiment_input)[0]

    return category, sentiment


# Session storage
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []


# Sidebar
st.sidebar.title("🎓 Student Feedback System")

page = st.sidebar.radio(
    "Navigation",
    ["Student Dashboard", "Admin Dashboard"]
)


# STUDENT DASHBOARD
if page == "Student Dashboard":

    st.title("🎓 Student Feedback Portal")
    st.write("Share your feedback about our college.")

    st.divider()

    student_name = st.text_input("Student Name")

    feedback = st.text_area(
        "Enter Your Feedback",
        placeholder="Write your feedback here...",
        height=150
    )

    if st.button("Submit Feedback", type="primary"):

        if not student_name.strip():
            st.warning("Please enter your name.")

        elif not feedback.strip():
            st.warning("Please enter your feedback.")

        else:
            category, sentiment = predict_feedback(feedback)

            record = {
                "Student": student_name,
                "Feedback": feedback,
                "Category": category,
                "Sentiment": sentiment
            }

            st.session_state.feedback_data.append(record)

            st.success("Feedback submitted successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Category", category)

            with col2:
                st.metric("Predicted Sentiment", sentiment)


# ADMIN DASHBOARD
elif page == "Admin Dashboard":

    st.title("📊 Admin Dashboard")

    password = st.sidebar.text_input(
        "Admin Password",
        type="password"
    )

    if password == "admin123":

        st.success("Admin login successful!")

        data = pd.DataFrame(st.session_state.feedback_data)

        if not data.empty:

            st.subheader("Feedback Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Feedback", len(data))

            with col2:
                st.metric(
                    "Categories",
                    data["Category"].nunique()
                )

            with col3:
                st.metric(
                    "Positive Feedback",
                    (data["Sentiment"] == "positive").sum()
                )

            st.divider()

            st.subheader("Feedback Category Distribution")

            category_counts = data["Category"].value_counts()

            st.bar_chart(category_counts)

            st.subheader("Sentiment Distribution")

            sentiment_counts = data["Sentiment"].value_counts()

            st.bar_chart(sentiment_counts)

            st.subheader("All Student Feedback")

            st.dataframe(data, use_container_width=True)

            csv = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Feedback Report",
                data=csv,
                file_name="student_feedback_report.csv",
                mime="text/csv"
            )

        else:
            st.info("No student feedback has been submitted yet.")

    else:
        st.info("Enter the admin password in the sidebar.")
