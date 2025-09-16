import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Set page config for a custom title and icon
st.set_page_config(page_title="Axis Guidelines AI", page_icon="📄", layout="wide")

# Load API key from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')  # Free tier, fast

# Load processed data
with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)
index = faiss.read_index('index.faiss')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Custom CSS for styling
st.markdown("""
<style>
.stApp { background-color: #e6f3ff; } /* Light blue background */
.stTextInput > div > div > input { background-color: #ffffff; border: 2px solid #004aad; } /* Blue border input */
.stButton > button { background-color: #004aad; color: white; } /* Blue button */
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation and branding
with st.sidebar:
    st.header("Axis Guidelines AI")
    st.markdown("Ask questions about our company guidelines, powered by your PDFs!")
    # Optional: Add a logo (upload to repo and uncomment)
    # st.image("logo.png", width=200)
    st.markdown("**How to use:**")
    st.markdown("- Enter a question about our guidelines.")
    st.markdown("- Click 'Get Answer' to see results.")
    st.markdown("- Check sources for reference.")
    st.markdown("**Example questions:**")
    st.markdown("- What are the content design principles?")
    st.markdown("- How should headings be written?")

# Main content
st.title("📄 Axis Guidelines AI Assistant")
st.markdown("Get instant answers from our company guidelines. Type your question below!")

# Input form with button
with st.form(key="question_form"):
    query = st.text_input("Your question:", placeholder="e.g., What are the content design principles?", help="Ask about policies in our PDFs.")
    submit_button = st.form_submit_button("Get Answer")
    clear_button = st.form_submit_button("Clear")

if submit_button and query:
    # Search for relevant chunks
    query_emb = embed_model.encode([query])
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb.astype('float32'), k=3)  # Top 3 chunks
    context = "\n\n".join([chunks[i] for i in I[0]])
    
    # Ask Gemini (temperature=0 for consistent answers)
    prompt = f"Answer based ONLY on this context from our guidelines:\n\n{context}\n\nQuestion: {query}\n\nGive a clear, concise answer."
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
    
    # Display results
    st.subheader("Answer")
    st.markdown(response.text, help="This answer is generated from your PDFs.")
    
    st.subheader("Sources")
    for i in I[0]:
        st.markdown(f"• {chunks[i][:150]}...")
    
    # Add a divider
    st.markdown("---")

# Help section (collapsible)
with st.expander("Need Help?"):
    st.markdown("""
    - **Updating PDFs**: Add/remove PDFs in `test_pdfs`, run `processor.py`, and push to GitHub.
    - **Contact**: Reach out to the admin for issues or new guideline PDFs.
    - **Tips**: Ask specific questions for best results (e.g., 'What is the tone for content design?').
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for the Axis team | Last updated: September 2025")