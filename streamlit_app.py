import streamlit as st
import requests

# Function to generate content using the Gemini API
def generate_content(prompt):
    url = "https://api.gemini.com/generate"  # Replace with the actual API endpoint
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",  # Replace with your actual API key
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 150  # Adjust as needed
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()['text']  # Adjust based on API response structure
    else:
        st.error(f"Error: {response.status_code} - {response.json().get('error', {}).get('message', 'Unknown error')}")
        return None

# Function to repurpose content
def repurpose_content(original_content, format_type):
    if format_type == "Blog Post":
        return f"## Blog Post\n\n{original_content}"
    elif format_type == "Social Media Post":
        return f"📣 {original_content[:100]}..."  # Shortened for social media
    elif format_type == "Email":
        return f"Subject: Exciting Update!\n\n{original_content}"
    return original_content

# Main Streamlit app function
def main():
    st.title("Content Repurposing Platform")
    
    # Input for the content generation prompt
    prompt = st.text_area("Enter your content prompt:")
    
    if st.button("Generate Content"):
        if prompt:
            generated_content = generate_content(prompt)
            if generated_content:
                st.subheader("Generated Content:")
                st.write(generated_content)

                # Options to repurpose the content
                format_type = st.selectbox("Select format to repurpose content:", 
                                            ["Blog Post", "Social Media Post", "Email"])
                
                if st.button("Repurpose Content"):
                    repurposed_content = repurpose_content(generated_content, format_type)
                    st.subheader(f"{format_type} Preview:")
                    st.write(repurposed_content)
        else:
            st.error("Please enter a prompt!")

if __name__ == "__main__":
    main()
