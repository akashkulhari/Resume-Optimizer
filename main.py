import streamlit as st
import os
import tempfile
import shutil
import base64
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_rag_completion(
    documents,
    query_text: str,
    job_title: str,
    job_description: str,
    # UPDATED: Using the latest 2026 multimodal embedding model
    embedding_model: str = "models/gemini-embedding-2-preview",
    # UPDATED: Using the current Gemini 3 series
    generative_model: str = "models/gemini-3-flash-preview"
) -> str:
    """Run RAG completion using Gemini models for resume optimization."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment.")

        # Initialize Models
        llm = GoogleGenAI(model=generative_model, api_key=api_key)
        embed_model = GoogleGenAIEmbedding(model_name=embedding_model, api_key=api_key)
        
        # Apply settings globally to LlamaIndex
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Step 1: Analyze the resume
        analysis_prompt = """
        Analyze this resume in detail. Focus on:
        1. Key skills and expertise
        2. Professional experience and achievements
        3. Education and certifications
        
        Provide a concise analysis in bullet points.
        """
        
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=5)
        
        # Initial analysis
        resume_analysis = query_engine.query(analysis_prompt)
        
        # Step 2: Optimization suggestions
        optimization_prompt = f"""
        Based on the resume analysis and job requirements, provide specific improvements.
        
        Resume Analysis: {resume_analysis}
        Job Title: {job_title}
        Job Description: {job_description}
        Optimization Request: {query_text}
        
        Provide a direct response in this format:
        ## Key Findings
        • [Alignment/Gaps]
        ## Specific Improvements
        • [Actionable suggestions]
        ## Action Items
        • [Immediate steps]
        """
        
        final_response = query_engine.query(optimization_prompt)
        return str(final_response)
        
    except Exception as e:
        return f"Optimization failed: {str(e)}"

def display_pdf_preview(pdf_file):
    """Display PDF preview in the sidebar."""
    try:
        base64_pdf = base64.b64encode(pdf_file.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
        st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Preview error: {str(e)}")

def main():
    st.set_page_config(page_title="Resume Optimizer", layout="wide")
    
    # Session state initialization
    if "messages" not in st.session_state: st.session_state.messages = []
    if "docs_loaded" not in st.session_state: st.session_state.docs_loaded = False
    if "temp_dir" not in st.session_state: st.session_state.temp_dir = None
    
    st.title("📝 Resume Optimizer")
    st.caption("Powered by Gemini 3 & Embedding 2")
    
    with st.sidebar:
        # Updated selection with 2026 models
        gen_model_choice = st.selectbox(
            "Select AI Intelligence",
            ["models/gemini-3-flash-preview", "models/gemini-3.1-pro-preview"],
            index=0
        )
        
        st.divider()
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        
        if uploaded_file:
            if st.session_state.temp_dir: shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = tempfile.mkdtemp()
            
            file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Indexing Resume..."):
                st.session_state.documents = SimpleDirectoryReader(st.session_state.temp_dir).load_data()
                st.session_state.docs_loaded = True
                st.success("Resume Loaded")
                display_pdf_preview(uploaded_file)

    # UI Columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Target Role")
        job_title = st.text_input("Job Title (e.g., Software Engineer)")
        job_description = st.text_area("Job Description", height=200)
        
        opt_type = st.selectbox("Goal", ["ATS Keyword Optimizer", "Experience Section Enhancer", "Professional Summary Crafter"])
        
        if st.button("Optimize Resume"):
            if not st.session_state.docs_loaded:
                st.error("Upload a resume first.")
            elif not job_description:
                st.error("Paste the job description.")
            else:
                with st.spinner("AI is analyzing..."):
                    response = run_rag_completion(
                        st.session_state.documents,
                        opt_type,
                        job_title,
                        job_description,
                        "models/gemini-embedding-2-preview",
                        gen_model_choice
                    )
                    st.session_state.messages.append({"content": response})

    with col2:
        st.subheader("Results")
        for msg in st.session_state.messages:
            st.markdown(msg["content"])

if __name__ == "__main__":
    main()