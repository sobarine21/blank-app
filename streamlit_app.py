import streamlit as st
import sqlite3
import random

# Initialize database connection
def init_db():
    try:
        conn = sqlite3.connect('community_feed.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            likes INTEGER DEFAULT 0,
            username TEXT NOT NULL
        )
        ''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )
        ''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS polls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            option_a TEXT NOT NULL,
            option_b TEXT NOT NULL,
            votes_a INTEGER DEFAULT 0,
            votes_b INTEGER DEFAULT 0
        )
        ''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS confessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        return conn, c
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return None, None

conn, c = init_db()

# Helper functions
def random_nickname():
    nicknames = ["Mysterious Dolphin", "Sneaky Ninja", "Curious Cat", "Wandering Wizard", "Clever Fox"]
    return random.choice(nicknames)

def add_post(content):
    try:
        username = random_nickname()
        c.execute('INSERT INTO posts (content, username) VALUES (?, ?)', (content, username))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error adding post: {e}")

def get_posts():
    try:
        c.execute('SELECT id, content, created_at, likes, username FROM posts ORDER BY created_at DESC')
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error retrieving posts: {e}")
        return []

def add_like(post_id):
    try:
        c.execute('UPDATE posts SET likes = likes + 1 WHERE id = ?', (post_id,))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error adding like: {e}")

def add_comment(post_id, content):
    try:
        c.execute('INSERT INTO comments (post_id, content) VALUES (?, ?)', (post_id, content))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error adding comment: {e}")

def get_comments(post_id):
    try:
        c.execute('SELECT content, created_at FROM comments WHERE post_id = ? ORDER BY created_at DESC', (post_id,))
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error retrieving comments: {e}")
        return []

def add_poll(question, option_a, option_b):
    try:
        c.execute('INSERT INTO polls (question, option_a, option_b) VALUES (?, ?, ?)', (question, option_a, option_b))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error adding poll: {e}")

def get_polls():
    try:
        c.execute('SELECT id, question, option_a, option_b, votes_a, votes_b FROM polls')
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error retrieving polls: {e}")
        return []

def vote_poll(poll_id, option):
    try:
        if option == 'A':
            c.execute('UPDATE polls SET votes_a = votes_a + 1 WHERE id = ?', (poll_id,))
        else:
            c.execute('UPDATE polls SET votes_b = votes_b + 1 WHERE id = ?', (poll_id,))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error voting in poll: {e}")

def add_confession(content):
    try:
        c.execute('INSERT INTO confessions (content) VALUES (?)', (content,))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error adding confession: {e}")

def get_confessions():
    try:
        c.execute('SELECT content, created_at FROM confessions ORDER BY created_at DESC')
        return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error retrieving confessions: {e}")
        return []

# Streamlit app layout
st.set_page_config(page_title="Anonymous Community Feed", page_icon="🌐", layout="wide")
st.title("🌐 Anonymous Community Feed")

# Sidebar for creating posts
st.sidebar.header("📝 Share Something")
post_content = st.sidebar.text_area("What's on your mind?", max_chars=280)
submit_button = st.sidebar.button("Post")

if submit_button and post_content:
    add_post(post_content)
    st.sidebar.success("🎉 Your post has been shared!")

# Display posts
st.subheader("📢 Community Feed")
posts = get_posts()

if posts:
    for post_id, content, created_at, likes, username in posts:
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; margin-bottom: 10px;'><strong>{username}:</strong> {content}</div>", unsafe_allow_html=True)
        st.write(f"📅 *Posted on {created_at}* - 👍 {likes} Likes")

        # Like functionality
        like_key = f"like_button_{post_id}"
        if st.button(f"👻 Like ({likes})", key=like_key):  # Unique key
            add_like(post_id)
            st.success("✨ Someone liked this!")

        # Comment functionality
        comment_key = f"comment_input_{post_id}"
        comment_text = st.text_input(f"Add a comment for post {post_id}", key=comment_key)  # Unique key
        if st.button("💬 Submit Comment", key=f"submit_comment_{post_id}"):
            if comment_text:
                add_comment(post_id, comment_text)
                st.success("🗨️ Your comment has been added!")
            else:
                st.error("Please enter a comment.")

        # Display comments
        comments = get_comments(post_id)
        for comment, created_at in comments:
            st.write(f"🗨️ {comment} (Posted on {created_at})")
else:
    st.info("No posts yet. Be the first to share!")

# Confession Box
st.sidebar.header("🤫 Confession Box")
confession_content = st.sidebar.text_area("What's your secret?", max_chars=280)
if st.sidebar.button("Share Confession"):
    if confession_content:
        add_confession(confession_content)
        st.sidebar.success("🤐 Your confession has been shared!")
    else:
        st.sidebar.error("Please enter a confession.")

# Display Confessions
st.subheader("🤫 Confessions")
confessions = get_confessions()
if confessions:
    for content, created_at in confessions:
        st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; margin-bottom: 10px;'>{content}</div>", unsafe_allow_html=True)
        st.write(f"📅 *Confessed on {created_at}*")
else:
    st.info("No confessions yet. Be the first to share!")

# Polls Section
st.sidebar.header("📊 Create a Poll")
poll_question = st.sidebar.text_input("Poll Question")
poll_option_a = st.sidebar.text_input("Option A")
poll_option_b = st.sidebar.text_input("Option B")
if st.sidebar.button("Create Poll"):
    if poll_question and poll_option_a and poll_option_b:
        add_poll(poll_question, poll_option_a, poll_option_b)
        st.sidebar.success("🗳️ Your poll has been created!")
    else:
        st.sidebar.error("Please fill in all poll fields.")

# Display polls
st.subheader("📊 Community Polls")
polls = get_polls()
if polls:
    for poll_id, question, option_a, option_b, votes_a, votes_b in polls:
        st.write(f"**{question}**")
        st.write(f"1️⃣ {option_a} - {votes_a} Votes")
        st.write(f"2️⃣ {option_b} - {votes_b} Votes")
        vote_key = f"poll_radio_{poll_id}"
        vote_option = st.radio("Choose an option:", (option_a, option_b), key=vote_key)  # Unique key
        if st.button("Vote", key=f"vote_poll_button_{poll_id}"):  # Unique key
            vote_poll(poll_id, 'A' if vote_option == option_a else 'B')
            st.success("✅ Your vote has been recorded!")
else:
    st.info("No polls yet. Be the first to create one!")

# Close the database connection when done
if conn:
    conn.close()
