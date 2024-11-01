# Required Libraries
import streamlit as st
import pandas as pd
import random
import string

# Mystery Names Generator
def generate_mystery_name():
    adjectives = ["Bold", "Mysterious", "Silent", "Unknown", "Ghost"]
    nouns = ["User", "Poster", "Writer", "Voice", "Mind"]
    return f"{random.choice(adjectives)}_{random.choice(nouns)}"

# Initialize Session State
if 'posts' not in st.session_state:
    st.session_state.posts = []

if 'username' not in st.session_state:
    st.session_state.username = generate_mystery_name()

# Streamlit App
st.title("Anonymous Community Feed")

# User can change their mystery name
def change_username():
    st.session_state.username = generate_mystery_name()

st.button("Change Mystery Name", on_click=change_username)

# Display Current Mystery Name
st.write(f"Your Mystery Name: {st.session_state.username}")

# Text Input for New Posts
new_post = st.text_input("Make a new post:")

# Submit New Post
def submit_post():
    st.session_state.posts.append({"username": st.session_state.username, "post": new_post})
    st.session_state.posts = sorted(st.session_state.posts, key=lambda x: random.random())  # Shuffle posts for anonymity

if st.button("Post"):
    submit_post()

# Display Community Feed
st.header("Community Feed")
for post in st.session_state.posts:
    st.write(f"{post['username']}: {post['post']}")
