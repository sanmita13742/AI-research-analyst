import streamlit as st
import fitz  # PyMuPDF
import re
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

# Load models once (cached)
@st.cache_resource
def load_models():
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        nlp = spacy.load('en_core_web_sm')
        return sentence_model, nlp
    except:
        st.error("Please install required models: pip install sentence-transformers spacy")
        st.stop()

def parse_structured_output(text: str) -> Dict[str, str]:
    """Parse structured output from API responses"""
    sections = {}
    current_section = None
    current_content = []
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers (uppercase with colon)
        if re.match(r'^[A-Z_]+:', line):
            # Save previous section
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line.split(':')[0]
            current_content = []
        elif current_section:
            current_content.append(line)
    
    # Save last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

def render_markdown_section(content: str, title: str) -> None:
    """Render a section with proper markdown formatting"""
    if not content or content.strip() == "No results":
        st.info(f"No {title.lower()} available yet.")
        return
    
    # Check for API errors
    if "Failed after 3 attempts" in content or "API Error" in content or "ERROR" in content:
        st.error(f"❌ {title} failed. Please check your API key and try again.")
        with st.expander("🔍 Debug Information", expanded=False):
            st.code(content)
        return
    
    # Try to parse structured content
    sections = parse_structured_output(content)
    
    if sections:
        # Render structured content
        for section_name, section_content in sections.items():
            with st.expander(f"📋 {section_name.replace('_', ' ').title()}", expanded=True):
                st.markdown(section_content)
    else:
        # Render as regular markdown
        st.markdown(content)

class AIResearchAnalyzer:
    def __init__(self):
        self.sentence_model, self.nlp = load_models()
        self.groq_api_key = None
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF using PyMuPDF"""
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
            return ""
    
    def call_groq_api(self, prompt: str, max_tokens: int = 1500) -> str:
        """Call Groq API with error handling and retries"""
        if not self.groq_api_key:
            return "ERROR: Please provide Groq API key"
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Truncate prompt if too long (Groq has limits)
        if len(prompt) > 32000:  # Leave some buffer
            prompt = prompt[:32000] + "\n\n[Content truncated due to length limits]"
        
        data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        for attempt in range(3):  # 3 retry attempts
            try:
                response = requests.post(url, headers=headers, json=data, timeout=60)  # Increased timeout
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"]
                    if result and result.strip():
                        return result
                    else:
                        return "ERROR: Empty response from API"
                elif response.status_code == 429:
                    st.warning(f"Rate limited, attempt {attempt + 1}/3. Waiting...")
                    time.sleep(3)  # Longer backoff
                    continue
                elif response.status_code == 401:
                    return "ERROR: Invalid API key. Please check your Groq API key."
                elif response.status_code == 400:
                    return f"ERROR: Bad request - {response.text}"
                else:
                    return f"API Error: {response.status_code} - {response.text}"
            except requests.exceptions.Timeout:
                if attempt == 2:  # Last attempt
                    return "ERROR: Request timed out after 60 seconds"
                time.sleep(2)
            except Exception as e:
                if attempt == 2:  # Last attempt
                    return f"ERROR: {str(e)}"
                time.sleep(2)
        
        return "ERROR: Failed after 3 attempts - please check your internet connection and API key"

    def chain_prompt_1_document_analysis(self, text: str) -> str:
        """CHAIN 1: Comprehensive document structure analysis"""
        
        prompt = f"""
        You are an expert research paper analyzer. Analyze this research paper text and identify ALL sections systematically.

        TASK: Carefully read through this research paper and identify:
        1. The ABSTRACT section (regardless of what it's called)
        2. ALL methodology-related content (from ANY section)
        3. The main research objectives
        4. The paper structure and section headings

        PAPER TEXT:
        {text[:8000]}  # Limit to first 8000 chars to stay within token limits

        OUTPUT FORMAT (be extremely precise):
        ABSTRACT_CONTENT: [Extract the complete abstract/summary, even if called "Summary", "Overview", etc.]

        METHODOLOGY_CONTENT: [Extract ALL methodology content from ANYWHERE in the paper - this could be in "Methods", "Methodology", "Approach", "Experimental Setup", "Materials and Methods", "Procedure", "Implementation", "Design", or embedded in other sections]

        RESEARCH_OBJECTIVES: [What are the main goals/objectives of this research?]

        SECTION_HEADINGS: [List all major section headings you found]

        CONFIDENCE_SCORES: 
        - Abstract extraction: [1-10]
        - Methodology extraction: [1-10]
        - Overall paper understanding: [1-10]

        Be thorough - methodology information might be scattered across multiple sections!
        """
        
        return self.call_groq_api(prompt, max_tokens=2000)

    def chain_prompt_2_methodology_refinement(self, initial_analysis: str, text: str) -> str:
        """CHAIN 2: Deep methodology extraction and refinement"""
        
        prompt = f"""
        You are a methodology extraction specialist. Based on the initial analysis, perform a DEEPER extraction of methodology.

        INITIAL ANALYSIS:
        {initial_analysis}

        TASK: Now do a SECOND PASS through the paper to find ANY missed methodology details:
        1. Look for experimental procedures in Results sections
        2. Check for methodological details in Introduction/Background
        3. Find implementation details in Discussion/Conclusion
        4. Look for data collection methods, analysis techniques, tools used
        5. Identify statistical methods, algorithms, frameworks

        PAPER TEXT (focusing on missed sections):
        {text[4000:12000]}  # Different section of the paper

        OUTPUT FORMAT:
        REFINED_METHODOLOGY: [Complete, comprehensive methodology combining initial findings + new discoveries]

        METHODOLOGY_CATEGORIES:
        - Data Collection: [How data was collected]
        - Analysis Methods: [Statistical/computational methods used]
        - Tools/Software: [What tools, software, frameworks]
        - Experimental Design: [How the study was designed]
        - Validation Methods: [How results were validated]

        MISSED_DETAILS: [What additional methodology details were found in this second pass]

        Only output content that actually exists in the paper. Don't make assumptions.
        """
        
        return self.call_groq_api(prompt, max_tokens=2000)

    def chain_prompt_3_user_project_analysis(self, user_responses: Dict[str, str]) -> str:
        """CHAIN 3: Analyze and structure user's project details"""
        
        user_text = "\n".join([f"{k}: {v}" for k, v in user_responses.items() if v.strip()])
        
        prompt = f"""
        You are a research project analyzer. Analyze the user's project description and create a structured analysis.

        USER'S PROJECT RESPONSES:
        {user_text}

        TASK: Create a comprehensive analysis of the user's project:

        OUTPUT FORMAT:
        USER_ABSTRACT: [Create a clear, concise abstract of the user's project based on their responses]

        USER_METHODOLOGY: [Detail the user's planned methodology, approach, and methods]

        PROJECT_STRUCTURE:
        - Research Question: [What question are they trying to answer?]
        - Hypothesis: [What do they expect to find?]
        - Data Sources: [Where will data come from?]
        - Analysis Plan: [How will they analyze the data?]
        - Expected Outcomes: [What are they hoping to achieve?]

        CLARITY_ASSESSMENT:
        - Project clarity: [1-10]
        - Methodology completeness: [1-10]
        - Feasibility indicators: [List potential issues or strengths]

        Be thorough but only use information the user actually provided.
        """
        
        return self.call_groq_api(prompt, max_tokens=1500)

    def chain_prompt_5_scoring_evaluation(self, paper_analysis: str, user_analysis: str) -> str:
        """CHAIN 5: Final scoring and recommendations based on paper analysis and user project"""
        
        # Check if inputs are valid
        if not paper_analysis or "ERROR" in paper_analysis or "Failed" in paper_analysis:
            return "ERROR: Paper analysis not available for evaluation"
        
        if not user_analysis or "ERROR" in user_analysis or "Failed" in user_analysis:
            return "ERROR: User project analysis not available for evaluation"
        
        # Truncate inputs more aggressively to prevent API failures
        paper_truncated = paper_analysis[:8000] if len(paper_analysis) > 8000 else paper_analysis
        user_truncated = user_analysis[:8000] if len(user_analysis) > 8000 else user_analysis
        
        prompt = f"""
        You are a research evaluation expert. Evaluate the user's research project and provide comprehensive scoring.

        PAPER ANALYSIS (Key Points):
        {paper_truncated}

        USER PROJECT:
        {user_truncated}

        TASK: Provide detailed evaluation with scores and recommendations.

        OUTPUT FORMAT:
        DETAILED_SCORES:
        - Novelty/Innovation: [X/10] - [Brief explanation]
        - Methodological Rigor: [X/10] - [Brief explanation]
        - Feasibility: [X/10] - [Brief explanation]
        - Impact Potential: [X/10] - [Brief explanation]
        - Technical Soundness: [X/10] - [Brief explanation]

        OVERALL_RATING: [X/10] - [Overall assessment]

        PROJECT_STRENGTHS:
        - [Key strength 1]
        - [Key strength 2]
        - [Key strength 3]

        AREAS_FOR_IMPROVEMENT:
        - [Area 1]
        - [Area 2]
        - [Area 3]

        PRIORITY_RECOMMENDATIONS:
        1. [Most important recommendation]
        2. [Second recommendation]
        3. [Third recommendation]

        IMPROVEMENT_ROADMAP:
        - Immediate: [What to do now]
        - Short-term: [Next 1-2 months]
        - Long-term: [Future considerations]

        Be concise and actionable. Focus on the user's project quality.
        """
        
        return self.call_groq_api(prompt, max_tokens=1500)

    def test_api_connection(self) -> bool:
        """Test if the API key is working"""
        if not self.groq_api_key:
            return False
            
        test_prompt = "Please respond with 'API working' if you can see this message."
        
        try:
            result = self.call_groq_api(test_prompt, max_tokens=50)
            return "API working" in result or "ERROR" not in result
        except:
            return False

    def run_analysis_chain(self, paper_text: str, user_responses: Dict[str, str]) -> Dict[str, str]:
        """Execute the complete 4-step analysis chain"""
        
        results = {}
        
        # Test API connection first
        if not self.test_api_connection():
            st.error("❌ API connection failed. Please check your Groq API key.")
            return {
                'step1': 'ERROR: API connection failed',
                'step2': 'ERROR: API connection failed', 
                'step3': 'ERROR: API connection failed',
                'step4': 'ERROR: API connection failed'
            }
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Chain 1: Document Analysis
        status_text.text("🔍 Step 1/4: Analyzing document structure and content...")
        progress_bar.progress(25)
        results['step1'] = self.chain_prompt_1_document_analysis(paper_text)
        
        # Check if step 1 failed
        if "ERROR" in results['step1']:
            st.error(f"❌ Step 1 failed: {results['step1']}")
            return results
        
        # Chain 2: Methodology Refinement
        status_text.text("🔬 Step 2/4: Deep methodology extraction...")
        progress_bar.progress(50)
        results['step2'] = self.chain_prompt_2_methodology_refinement(results['step1'], paper_text)
        
        # Check if step 2 failed
        if "ERROR" in results['step2']:
            st.error(f"❌ Step 2 failed: {results['step2']}")
            return results
        
        # Chain 3: User Project Analysis
        status_text.text("👤 Step 3/4: Analyzing your project structure...")
        progress_bar.progress(75)
        results['step3'] = self.chain_prompt_3_user_project_analysis(user_responses)
        
        # Check if step 3 failed
        if "ERROR" in results['step3']:
            st.error(f"❌ Step 3 failed: {results['step3']}")
            return results
        
        # Chain 4: Final Scoring
        status_text.text("📊 Step 4/4: Final evaluation and scoring...")
        progress_bar.progress(100)
        results['step4'] = self.chain_prompt_5_scoring_evaluation(results['step1'], results['step3'])
        
        # Check if step 4 failed
        if "ERROR" in results['step4']:
            st.error(f"❌ Step 4 failed: {results['step4']}")
            st.info("💡 Tip: Try reducing the complexity of your project description or check your API key.")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results

    def ask_research_questions(self) -> Dict[str, str]:
        """Enhanced interactive questionnaire with better UX and default values"""
        st.subheader("📝 Tell us about your research project")
        st.markdown("*The AI will analyze your responses comprehensively - be as detailed as possible.*")
        
        # Default values for ease of use
        default_values = {
            "research_problem": "I want to analyze the impact of machine learning algorithms on data processing efficiency in research applications.",
            "research_question": "How do different machine learning approaches affect the accuracy and speed of research data analysis?",
            "methodology_approach": "Mixed-methods approach combining quantitative analysis with qualitative assessment of algorithm performance.",
            "data_collection": "Primary data from research datasets, secondary data from academic databases, and experimental results from algorithm testing.",
            "analysis_methods": "Statistical analysis (regression, ANOVA), machine learning evaluation metrics, and comparative performance analysis.",
            "tools_software": "Python, scikit-learn, TensorFlow, R, SPSS, and Jupyter notebooks for analysis and visualization.",
            "sample_population": "Research datasets from academic institutions, sample size of 1000+ data points across multiple domains.",
            "innovation_novelty": "Novel combination of traditional statistical methods with advanced ML techniques for research applications.",
            "expected_outcomes": "Improved understanding of ML algorithm performance in research contexts and recommendations for optimal methodology selection.",
            "challenges_limitations": "Data quality issues, computational resource constraints, and potential bias in algorithm selection.",
            "timeline_resources": "6-month timeline with access to university computing resources and academic datasets."
        }
        
        user_responses = {}
        
        # Create a more organized form with interactive elements
        with st.form("research_project_form"):
            st.markdown("### 🎯 Research Details")
            
            # Quick fill options
            col_quick, col_clear = st.columns(2)
            with col_quick:
                if st.form_submit_button("🚀 Load Example Project", type="secondary"):
                    st.session_state.use_defaults = True
            with col_clear:
                if st.form_submit_button("🗑️ Clear All Fields", type="secondary"):
                    st.session_state.use_defaults = False
            
            # Use defaults if requested
            use_defaults = st.session_state.get('use_defaults', False)
            
            col1, col2 = st.columns(2)
            with col1:
                user_responses["research_problem"] = st.text_area(
                    "Research Problem", 
                    value=default_values["research_problem"] if use_defaults else "",
                    key="research_problem", 
                    height=100,
                    help="Describe the specific problem you're trying to solve",
                    placeholder="e.g., I want to analyze the impact of..."
                )
                user_responses["research_question"] = st.text_area(
                    "Research Question/Hypothesis", 
                    value=default_values["research_question"] if use_defaults else "",
                    key="research_question", 
                    height=80,
                    help="What is your main research question?",
                    placeholder="e.g., How do different approaches affect..."
                )
                user_responses["methodology_approach"] = st.text_area(
                    "Methodology Approach", 
                    value=default_values["methodology_approach"] if use_defaults else "",
                    key="methodology_approach", 
                    height=80,
                    help="What methodology will you use?",
                    placeholder="e.g., Quantitative, qualitative, mixed-methods..."
                )
                user_responses["data_collection"] = st.text_area(
                    "Data Collection", 
                    value=default_values["data_collection"] if use_defaults else "",
                    key="data_collection", 
                    height=80,
                    help="How will you collect data?",
                    placeholder="e.g., Surveys, experiments, databases..."
                )
                user_responses["analysis_methods"] = st.text_area(
                    "Analysis Methods", 
                    value=default_values["analysis_methods"] if use_defaults else "",
                    key="analysis_methods", 
                    height=80,
                    help="What analysis techniques will you use?",
                    placeholder="e.g., Statistical analysis, ML algorithms..."
                )
            
            with col2:
                user_responses["tools_software"] = st.text_area(
                    "Tools & Software", 
                    value=default_values["tools_software"] if use_defaults else "",
                    key="tools_software", 
                    height=80,
                    help="What tools will you use?",
                    placeholder="e.g., Python, R, SPSS, specialized software..."
                )
                user_responses["sample_population"] = st.text_area(
                    "Sample/Population", 
                    value=default_values["sample_population"] if use_defaults else "",
                    key="sample_population", 
                    height=80,
                    help="Describe your target population",
                    placeholder="e.g., Students, professionals, datasets..."
                )
                user_responses["innovation_novelty"] = st.text_area(
                    "Innovation/Novelty", 
                    value=default_values["innovation_novelty"] if use_defaults else "",
                    key="innovation_novelty", 
                    height=80,
                    help="What makes your approach unique?",
                    placeholder="e.g., Novel combination, new methodology..."
                )
                user_responses["expected_outcomes"] = st.text_area(
                    "Expected Outcomes", 
                    value=default_values["expected_outcomes"] if use_defaults else "",
                    key="expected_outcomes", 
                    height=80,
                    help="What outcomes do you expect?",
                    placeholder="e.g., Improved understanding, new insights..."
                )
                user_responses["challenges_limitations"] = st.text_area(
                    "Challenges & Limitations", 
                    value=default_values["challenges_limitations"] if use_defaults else "",
                    key="challenges_limitations", 
                    height=80,
                    help="What challenges do you anticipate?",
                    placeholder="e.g., Data quality, resource constraints..."
                )
                user_responses["timeline_resources"] = st.text_area(
                    "Timeline & Resources", 
                    value=default_values["timeline_resources"] if use_defaults else "",
                    key="timeline_resources", 
                    height=80,
                    help="What's your timeline and resources?",
                    placeholder="e.g., 6 months, university resources..."
                )
            
            # Form submit button
            submitted = st.form_submit_button("💾 Save Project Details", type="secondary")
            
            if submitted:
                filled_responses = {k: v for k, v in user_responses.items() if v.strip()}
                if len(filled_responses) >= 5:
                    st.success(f"✅ Project details saved! ({len(filled_responses)}/11 fields completed)")
                    st.session_state.user_responses = user_responses
                else:
                    st.warning(f"Please fill at least 5 fields ({len(filled_responses)}/5 completed)")
        
        # Return saved responses or current form data
        return st.session_state.get('user_responses', user_responses)

def main():
    st.set_page_config(
        page_title="AI Research Analyst Pro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: #333;
        font-weight: 500;
    }
    .metric-card h4 {
        color: #495057;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card p {
        color: #6c757d;
        margin: 0.3rem 0;
        font-size: 0.95rem;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(23, 162, 184, 0.2);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
    }
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
    }
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .progress-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Research Analyst Pro</h1>
        <p><strong>Advanced Chain-of-Prompts Analysis - Zero Manual Processing!</strong></p>
        <p><em>The AI handles everything - from document analysis to methodology extraction to final evaluation.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = AIResearchAnalyzer()
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("⚙️ Setup & Configuration")
        
        # API Key section
        with st.expander("🔑 API Configuration", expanded=True):
            groq_key = st.text_input(
                "Groq API Key (Free)", 
                value=analyzer.groq_api_key,
                type="password", 
                help="Get free API key from console.groq.com",
                placeholder="Enter your Groq API key..."
            )
            if groq_key:
                analyzer.groq_api_key = groq_key
                
                # Test API connection
                col_test1, col_test2 = st.columns([2, 1])
                with col_test1:
                    st.success("✅ API key configured!")
                with col_test2:
                    if st.button("🧪 Test API", type="secondary"):
                        with st.spinner("Testing API connection..."):
                            if analyzer.test_api_connection():
                                st.success("✅ API connection successful!")
                            else:
                                st.error("❌ API connection failed!")
            else:
                st.warning("⚠️ API key required for analysis")
        
        # Analysis progress
        if 'analysis_results' in st.session_state:
            st.markdown("### 📊 Analysis Status")
            st.success("✅ Analysis Complete!")
            
            # Quick metrics with better styling
            results = st.session_state.analysis_results
            st.markdown("""
            <div class="metric-card">
                <h4>📈 Analysis Summary</h4>
                <p>• Document analyzed</p>
                <p>• Methodology extracted</p>
                <p>• Project compared</p>
                <p>• Recommendations generated</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis chain info
        st.markdown("### 🔄 Analysis Chain")
        st.markdown("""
        1. **📋 Document Structure Analysis**
        2. **🔬 Deep Methodology Extraction** 
        3. **👤 User Project Analysis**
        4. **🏆 Final Scoring & Recommendations**
        """)
        
        # Features
        st.markdown("### ✨ Key Features")
        st.markdown("""
        - 🤖 AI-powered methodology extraction
        - 🔍 Multi-pass document analysis
        - 📊 Structured output parsing
        - 🎯 Comprehensive comparison
        - 📈 Detailed scoring system
        - 📄 Markdown-formatted results
        """)
        
        # Quick actions
        if 'analysis_results' in st.session_state:
            st.markdown("### 🚀 Quick Actions")
            if st.button("🔄 Run New Analysis", use_container_width=True):
                # Clear previous results
                for key in ['analysis_results', 'paper_text', 'user_responses']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Main interface
    if 'analysis_results' not in st.session_state:
        # Input section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📄 Upload Research Paper")
            
            uploaded_file = st.file_uploader(
                "Choose a PDF file", 
                type=['pdf'],
                help="Upload a research paper in PDF format"
            )
            
            if uploaded_file:
                with st.spinner("Extracting text from PDF..."):
                    text = analyzer.extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ PDF Successfully Processed!</h4>
                            <p><strong>Characters extracted:</strong> {:,}</p>
                            <p><strong>File:</strong> {}</p>
                        </div>
                        """.format(len(text), uploaded_file.name), unsafe_allow_html=True)
                        
                        st.session_state.paper_text = text
                        
                        # Enhanced preview
                        with st.expander("📖 Document Preview", expanded=False):
                            st.markdown("**First 500 characters:**")
                            st.text(text[:500] + "..." if len(text) > 500 else text)
                            
                            # Document stats
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Characters", f"{len(text):,}")
                            with col_b:
                                st.metric("Words", f"{len(text.split()):,}")
                            with col_c:
                                st.metric("Lines", f"{len(text.splitlines()):,}")
        
        with col2:
            st.header("🎯 Your Research Project")
            
            if 'paper_text' in st.session_state:
                user_responses = analyzer.ask_research_questions()
                
                # Check completion and show analysis button
                filled_responses = {k: v for k, v in user_responses.items() if v.strip()}
                completion_rate = len(filled_responses) / 11 * 100
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>📊 Project Completion</h4>
                    <p><strong>Fields completed:</strong> {len(filled_responses)}/11</p>
                    <p><strong>Completion rate:</strong> {completion_rate:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                if len(filled_responses) >= 5:
                    if st.button("🚀 Start AI Analysis Chain", type="primary", use_container_width=True):
                        if analyzer.groq_api_key:
                            # Run analysis
                            results = analyzer.run_analysis_chain(
                                st.session_state.paper_text, 
                                user_responses
                            )
                            
                            # Store results
                            st.session_state.analysis_results = results
                            st.session_state.analysis_timestamp = datetime.now()
                            st.success("✅ Analysis complete!")
                            st.rerun()
                        else:
                            st.error("❌ Please provide your Groq API key in the sidebar!")
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>⚠️ Incomplete Project Details</h4>
                        <p>Please complete at least 5 fields ({len(filled_responses)}/5 completed)</p>
                        <p>You can use the "Load Example Project" button to get started quickly!</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <h4>👆 Upload Required</h4>
                    <p>Please upload a research paper first to begin the analysis.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display results if available
    if 'analysis_results' in st.session_state:
        st.header("📊 AI Analysis Results")
        
        # Analysis metadata
        if 'analysis_timestamp' in st.session_state:
            st.caption(f"Analysis completed on: {st.session_state.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = st.session_state.analysis_results
        
        # Create tabs for different analysis steps
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Document Analysis", 
            "🔬 Methodology", 
            "👤 Your Project", 
            "🏆 Final Scores"
        ])
        
        with tab1:
            st.subheader("📋 Document Structure Analysis")
            render_markdown_section(results.get('step1', 'No results'), "Document Analysis")
        
        with tab2:
            st.subheader("🔬 Refined Methodology Extraction")
            render_markdown_section(results.get('step2', 'No results'), "Methodology")
        
        with tab3:
            st.subheader("👤 Your Project Analysis")
            render_markdown_section(results.get('step3', 'No results'), "Project Analysis")
        
        with tab4:
            st.subheader("🏆 Final Evaluation & Recommendations")
            render_markdown_section(results.get('step4', 'No results'), "Final Evaluation")
            
            # Enhanced download section
            st.markdown("---")
            st.subheader("📥 Export Results")
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # Download full report
                full_report = f"""
# AI Research Analysis Report
*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Document Analysis
{results.get('step1', 'N/A')}

## Methodology Extraction
{results.get('step2', 'N/A')}

## User Project Analysis
{results.get('step3', 'N/A')}

## Final Evaluation
{results.get('step4', 'N/A')}
"""
                
                st.download_button(
                    label="📄 Download Full Report (Markdown)",
                    data=full_report,
                    file_name=f"ai_research_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col_d2:
                # Download as JSON
                json_report = {
                    "timestamp": datetime.now().isoformat(),
                    "document_analysis": results.get('step1', ''),
                    "methodology": results.get('step2', ''),
                    "project_analysis": results.get('step3', ''),
                    "final_evaluation": results.get('step4', '')
                }
                
                st.download_button(
                    label="📊 Download JSON Data",
                    data=json.dumps(json_report, indent=2),
                    file_name=f"ai_research_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()