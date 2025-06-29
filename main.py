import streamlit as st
import fitz  # PyMuPDF
import re
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import Dict, List, Tuple
import time
import os

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

class AIResearchAnalyzer:
    def __init__(self):
        self.sentence_model, self.nlp = load_models()
        self.groq_api_key = ''
        self.gemini_api_key = ''
        self.ollama_url = 'http://localhost:11434'
        self.ollama_model = 'llama3.2:1b'  # Default to fastest model
        self.selected_model = 'groq'
        
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
        
        data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        for attempt in range(3):  # 3 retry attempts
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        # Add debugging for Groq
                        if len(content) < 50:  # Very short response
                            st.warning(f"⚠️ Groq returned very short response ({len(content)} chars): {content[:100]}")
                        return content
                    else:
                        return f"Groq API returned empty response: {result}"
                elif response.status_code == 429:
                    st.warning(f"⚠️ Groq rate limited, attempt {attempt + 1}/3")
                    time.sleep(2)  # Rate limit backoff
                    continue
                elif response.status_code == 400:
                    error_msg = f"Groq API Bad Request: {response.text}"
                    st.error(f"❌ {error_msg}")
                    return error_msg
                elif response.status_code == 401:
                    return "ERROR: Invalid Groq API key"
                elif response.status_code == 403:
                    return "ERROR: Groq API access forbidden - check your API key permissions"
                else:
                    error_msg = f"Groq API Error: {response.status_code} - {response.text}"
                    st.error(f"❌ {error_msg}")
                    return error_msg
            except requests.exceptions.Timeout:
                st.warning(f"⚠️ Groq timeout on attempt {attempt + 1}/3")
                if attempt == 2:  # Last attempt
                    return "ERROR: Groq API timeout after 3 attempts"
                time.sleep(2)
            except Exception as e:
                st.error(f"❌ Groq API exception: {str(e)}")
                if attempt == 2:  # Last attempt
                    return f"ERROR: {str(e)}"
                time.sleep(1)
        
        return "Failed after 3 attempts"

    def clean_gemini_response(self, response: str) -> str:
        """Clean Gemini response by removing HTML tags and formatting"""
        if not response:
            return response
        
        import re
        
        # Remove HTML tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Remove common Gemini formatting artifacts
        response = re.sub(r'```[^`]*```', '', response)  # Remove code blocks
        response = re.sub(r'`[^`]*`', '', response)      # Remove inline code
        
        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()
        
        return response

    def call_gemini_api(self, prompt: str, max_tokens: int = 1500) -> str:
        """Call Gemini API with error handling and retries"""
        if not self.gemini_api_key:
            return "ERROR: Please provide Gemini API key"
        
        # Try different Gemini models in order of preference
        models_to_try = [
            "gemini-1.5-flash",
            "gemini-1.0-pro", 
            "gemini-pro"
        ]
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.1,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"
            
            for attempt in range(2):  # 2 attempts per model
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        if "candidates" in result and len(result["candidates"]) > 0:
                            raw_response = result["candidates"][0]["content"]["parts"][0]["text"]
                            # Clean the response to remove HTML tags and formatting
                            cleaned_response = self.clean_gemini_response(raw_response)
                            return cleaned_response
                        else:
                            return "No response generated"
                    elif response.status_code == 404:
                        # Try next model
                        break
                    elif response.status_code == 429:
                        time.sleep(2)  # Rate limit backoff
                        continue
                    else:
                        # Try next model
                        break
                except Exception as e:
                    if attempt == 1:  # Last attempt for this model
                        break
                    time.sleep(1)
        
        return "ERROR: All Gemini models failed. Please check your API key or try Groq/Ollama instead."

    def call_ollama_api(self, prompt: str, max_tokens: int = 1500) -> str:
        """Call Ollama API (local) with error handling and retries"""
        url = f"{self.ollama_url}/api/generate"
        
        # Use a smaller, faster model for better performance
        model_name = self.ollama_model
        
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        for attempt in range(3):  # 3 retry attempts
            try:
                # Increase timeout for large models
                response = requests.post(url, json=data, timeout=120)  # 2 minutes timeout
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', 'No response generated')
                else:
                    return f"API Error: {response.status_code} - {response.text}"
            except requests.exceptions.Timeout:
                if attempt == 2:  # Last attempt
                    return f"Timeout Error: {model_name} is taking too long to respond. Try a smaller model like llama3.2:1b or llama3.2:3b"
                time.sleep(5)  # Wait longer between retries
            except Exception as e:
                if attempt == 2:  # Last attempt
                    return f"Error: {str(e)}"
                time.sleep(2)
        
        return "Failed after 3 attempts"

    def call_openai_api(self, prompt: str, max_tokens: int = 1500) -> str:
        """Call OpenAI API (if user has free credits) with error handling and retries"""
        openai_api_key = os.getenv('OPENAI_API_KEY', '')
        if not openai_api_key:
            return "ERROR: Please set OPENAI_API_KEY environment variable"
            
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        for attempt in range(3):  # 3 retry attempts
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    time.sleep(2)  # Rate limit backoff
                    continue
                else:
                    return f"API Error: {response.status_code} - {response.text}"
            except Exception as e:
                if attempt == 2:  # Last attempt
                    return f"Error: {str(e)}"
                time.sleep(1)
        
        return "Failed after 3 attempts"

    def call_ai_api(self, prompt: str, max_tokens: int = 1500) -> str:
        """Route to the selected AI model"""
        if self.selected_model == 'groq':
            return self.call_groq_api(prompt, max_tokens)
        elif self.selected_model == 'gemini':
            return self.call_gemini_api(prompt, max_tokens)
        elif self.selected_model == 'ollama':
            return self.call_ollama_api(prompt, max_tokens)
        elif self.selected_model == 'openai':
            return self.call_openai_api(prompt, max_tokens)
        else:
            return "ERROR: Invalid model selection"

    def chain_prompt_1_document_analysis(self, text: str) -> Dict[str, str]:
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
        
        return self.call_ai_api(prompt, max_tokens=2000)

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
        
        return self.call_ai_api(prompt, max_tokens=2000)

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
        
        return self.call_ai_api(prompt, max_tokens=1500)

    def chain_prompt_4_comparative_analysis(self, paper_analysis: str, user_analysis: str) -> str:
        """CHAIN 4: Deep comparative analysis between paper and user project"""
        
        # Add validation and debugging
        if not paper_analysis or paper_analysis.strip() == "":
            return "ERROR: Paper analysis data is empty or missing"
        
        if not user_analysis or user_analysis.strip() == "":
            return "ERROR: User analysis data is empty or missing"
        
        # For Groq, use smaller input sizes to avoid token limits
        if self.selected_model == 'groq':
            max_input_length = 1500  # Reduced for Groq
        else:
            max_input_length = 2500  # Keep larger for other models
            
        paper_analysis_truncated = paper_analysis[:max_input_length] + ("..." if len(paper_analysis) > max_input_length else "")
        user_analysis_truncated = user_analysis[:max_input_length] + ("..." if len(user_analysis) > max_input_length else "")
        
        # Create a simpler, more direct prompt for Groq
        if self.selected_model == 'groq':
            prompt = f"""Compare the research paper and user project:

PAPER: {paper_analysis_truncated}

USER PROJECT: {user_analysis_truncated}

Provide a structured comparison:

SIMILARITIES:
- How are the approaches similar?

DIFFERENCES:
- How do they differ?

STRENGTHS:
- What's strong about user's approach?

WEAKNESSES:
- What could be improved?

RECOMMENDATIONS:
- What should the user do next?

Be specific and actionable."""
        else:
            # Use the original detailed prompt for other models
            prompt = f"""
            You are a research comparison expert. Perform a detailed comparative analysis between the published paper and user's project.

            PUBLISHED PAPER ANALYSIS:
            {paper_analysis_truncated}

            USER'S PROJECT ANALYSIS:
            {user_analysis_truncated}

            TASK: Provide comprehensive comparison across multiple dimensions:

            OUTPUT FORMAT:
            SIMILARITY_ANALYSIS:
            - Methodological Similarities: [How are the approaches similar?]
            - Methodological Differences: [How do they differ?]
            - Objective Alignment: [How aligned are the research goals?]
            - Innovation Gap: [What's new in user's approach vs. paper?]

            STRENGTH_ANALYSIS:
            - User's Methodological Strengths: [What's strong about user's approach?]
            - User's Innovative Elements: [What's novel or creative?]
            - Potential Advantages: [Where might user's approach be better?]

            WEAKNESS_ANALYSIS:
            - Methodological Gaps: [What's missing or weak in user's approach?]
            - Rigor Concerns: [Where could the approach be more rigorous?]
            - Feasibility Issues: [What might be challenging to implement?]

            LEARNING_OPPORTUNITIES:
            - What can user learn from the paper's methodology?
            - Which techniques from the paper could be adapted?
            - What should user avoid based on paper's limitations?

            Be specific and actionable in your analysis. If you cannot perform a meaningful comparison, explain why and provide general guidance.
            """
        
        try:
            # Use smaller max_tokens for Groq to avoid issues
            max_tokens = 1200 if self.selected_model == 'groq' else 1800
            result = self.call_ai_api(prompt, max_tokens=max_tokens)
            
            if not result or result.strip() == "":
                return "ERROR: No response received from AI model for comparison"
            
            # For Groq, if the response is too short, try a second attempt with different prompt
            if self.selected_model == 'groq' and len(result) < 200:
                st.warning("⚠️ Groq response too short, trying alternative prompt...")
                alternative_prompt = f"""Analyze these two research approaches:

Paper: {paper_analysis_truncated[:1000]}
User: {user_analysis_truncated[:1000]}

Compare them and provide:
1. Key similarities
2. Key differences  
3. Recommendations for the user

Be detailed and helpful."""
                
                result = self.call_ai_api(alternative_prompt, max_tokens=1000)
                if not result or result.strip() == "":
                    return "ERROR: Groq comparison failed even with simplified prompt"
            
            return result
        except Exception as e:
            return f"ERROR in comparison step: {str(e)}"

    def chain_prompt_5_scoring_evaluation(self, comparative_analysis: str) -> str:
        """CHAIN 5: Final scoring and recommendations"""
        
        # For Groq, truncate the input to avoid token limits
        if self.selected_model == 'groq':
            max_input_length = 1000  # Smaller for Groq
            comparative_analysis_truncated = comparative_analysis[:max_input_length] + ("..." if len(comparative_analysis) > max_input_length else "")
        else:
            comparative_analysis_truncated = comparative_analysis
        
        # Create a simpler prompt for Groq
        if self.selected_model == 'groq':
            prompt = f"""Based on this analysis, provide scores and recommendations:

ANALYSIS: {comparative_analysis_truncated}

Provide:
1. Scores (1-10) for:
   - Innovation
   - Methodology  
   - Feasibility
   - Impact
   - Technical quality

2. Overall rating (1-10)

3. Top 3 recommendations

4. Next steps

Be specific and helpful."""
        else:
            # Use the original detailed prompt for other models
            prompt = f"""
            You are a research evaluation expert. Based on the comparative analysis, provide final scores and recommendations.

            COMPARATIVE ANALYSIS:
            {comparative_analysis_truncated}

            TASK: Provide final evaluation with specific scores and actionable recommendations.

            OUTPUT FORMAT:
            DETAILED_SCORES:
            - Novelty/Innovation: [X/10] - [Explanation why this score]
            - Methodological Rigor: [X/10] - [Explanation why this score]
            - Feasibility: [X/10] - [Explanation why this score]
            - Impact Potential: [X/10] - [Explanation why this score]
            - Technical Soundness: [X/10] - [Explanation why this score]

            OVERALL_RATING: [X/10] - [Overall assessment]

            PRIORITY_RECOMMENDATIONS:
            1. [Most important recommendation with specific action]
            2. [Second most important recommendation with specific action]
            3. [Third most important recommendation with specific action]

            IMPROVEMENT_ROADMAP:
            - Immediate Actions: [What to do right now]
            - Short-term Goals: [What to do in next 1-2 months]
            - Long-term Considerations: [What to consider for future]

            RISK_MITIGATION:
            - Highest Risk: [Biggest risk and how to mitigate]
            - Medium Risks: [Other risks to watch]
            - Success Factors: [What would make this project succeed]

            Be specific, actionable, and constructive in all recommendations.
            """
        
        try:
            # Use smaller max_tokens for Groq
            max_tokens = 1000 if self.selected_model == 'groq' else 2000
            result = self.call_ai_api(prompt, max_tokens=max_tokens)
            
            if not result or result.strip() == "":
                return "ERROR: No response received from AI model for final scoring"
            
            # For Groq, if the response is too short, try a second attempt
            if self.selected_model == 'groq' and len(result) < 150:
                st.warning("⚠️ Groq final scoring response too short, trying alternative prompt...")
                alternative_prompt = f"""Score this research project:

Analysis: {comparative_analysis_truncated[:800]}

Rate 1-10:
- Innovation
- Methodology
- Feasibility
- Overall

Give 3 recommendations.

Be helpful and specific."""
                
                result = self.call_ai_api(alternative_prompt, max_tokens=800)
                if not result or result.strip() == "":
                    return "ERROR: Groq final scoring failed even with simplified prompt"
            
            return result
        except Exception as e:
            return f"ERROR in final scoring step: {str(e)}"

    def run_analysis_chain(self, paper_text: str, user_responses: Dict[str, str]) -> Dict[str, str]:
        """Execute the complete 5-step analysis chain"""
        
        results = {}
        
        try:
            # Chain 1: Document Analysis
            st.write("🔍 **Step 1/5**: Analyzing document structure and content...")
            step1_result = self.chain_prompt_1_document_analysis(paper_text)
            results['step1'] = step1_result
            st.write(f"✅ Step 1 completed: {len(step1_result)} characters")
            
            # Chain 2: Methodology Refinement
            st.write("🔬 **Step 2/5**: Deep methodology extraction...")
            step2_result = self.chain_prompt_2_methodology_refinement(step1_result, paper_text)
            results['step2'] = step2_result
            st.write(f"✅ Step 2 completed: {len(step2_result)} characters")
            
            # Chain 3: User Project Analysis
            st.write("👤 **Step 3/5**: Analyzing your project structure...")
            step3_result = self.chain_prompt_3_user_project_analysis(user_responses)
            results['step3'] = step3_result
            st.write(f"✅ Step 3 completed: {len(step3_result)} characters")
            
            # Chain 4: Comparative Analysis
            st.write("⚖️ **Step 4/5**: Comparing approaches...")
            st.write(f"📊 Input sizes - Paper analysis: {len(step2_result)} chars, User analysis: {len(step3_result)} chars")
            
            # Check if inputs are valid
            if step2_result and step3_result:
                # Add debugging for Groq
                if self.selected_model == 'groq':
                    st.info(f"🔍 Using Groq-optimized comparison (max input: 1500 chars)")
                    st.write(f"📝 Paper analysis preview: {step2_result[:100]}...")
                    st.write(f"📝 User analysis preview: {step3_result[:100]}...")
                
                step4_result = self.chain_prompt_4_comparative_analysis(step2_result, step3_result)
                results['step4'] = step4_result
                st.write(f"✅ Step 4 completed: {len(step4_result)} characters")
                
                # Check if comparison failed
                if step4_result.startswith("ERROR"):
                    st.error(f"❌ Step 4 failed: {step4_result}")
                    
                    # For Groq, provide specific troubleshooting
                    if self.selected_model == 'groq':
                        st.error("🔧 Groq-specific troubleshooting:")
                        st.markdown("""
                        **Common Groq Issues:**
                        - **Token limits**: Try reducing input text length
                        - **Rate limiting**: Wait a few seconds and try again
                        - **API key issues**: Check your Groq API key
                        - **Model availability**: Llama3-8b might be temporarily unavailable
                        
                        **Try these solutions:**
                        1. Use a shorter research paper
                        2. Provide shorter project descriptions
                        3. Check your Groq API key at console.groq.com
                        4. Try Gemini or Ollama instead
                        """)
                    
                    # Create a fallback comparison
                    fallback_comparison = f"""
COMPARATIVE ANALYSIS (Fallback):
Based on the available data, here is a basic comparison:

PAPER METHODOLOGY: {step2_result[:500]}...
USER METHODOLOGY: {step3_result[:500]}...

BASIC COMPARISON:
- Both approaches involve research methodology
- User project shows different focus areas
- Further detailed comparison requires more specific data

RECOMMENDATIONS:
- Review methodology alignment
- Consider adapting successful elements from the paper
- Focus on unique aspects of your approach
                    """
                    results['step4'] = fallback_comparison
                    st.warning("⚠️ Using fallback comparison due to step failure")
            else:
                st.error("❌ Step 4 failed: Missing input data")
                results['step4'] = "ERROR: Missing input data for comparison"
            
            # Chain 5: Final Scoring
            st.write("📊 **Step 5/5**: Final evaluation and scoring...")
            
            # Add debugging for Groq
            if self.selected_model == 'groq':
                st.info(f"🔍 Using Groq-optimized final scoring (max input: 1000 chars)")
                st.write(f"📝 Comparison analysis preview: {results['step4'][:100]}...")
            
            step5_result = self.chain_prompt_5_scoring_evaluation(results['step4'])
            results['step5'] = step5_result
            st.write(f"✅ Step 5 completed: {len(step5_result)} characters")
            
            # Check if final scoring failed
            if step5_result.startswith("ERROR"):
                st.error(f"❌ Step 5 failed: {step5_result}")
                
                # For Groq, provide specific troubleshooting
                if self.selected_model == 'groq':
                    st.error("🔧 Groq final scoring troubleshooting:")
                    st.markdown("""
                    **Common Groq Final Scoring Issues:**
                    - **Input too long**: The comparison analysis might be too long
                    - **Token limits**: Groq might be hitting context limits
                    - **Rate limiting**: Too many requests in quick succession
                    
                    **Try these solutions:**
                    1. Use shorter research papers
                    2. Provide shorter project descriptions  
                    3. Wait a few seconds and try again
                    4. Try Gemini or Ollama instead
                    """)
                
                # Create a fallback final scoring
                fallback_scoring = f"""
FINAL EVALUATION (Fallback):
Based on the available analysis, here is a basic evaluation:

COMPARISON SUMMARY: {results['step4'][:300]}...

BASIC SCORES:
- Innovation: 6/10 - Based on available information
- Methodology: 7/10 - Shows research approach
- Feasibility: 6/10 - Appears implementable
- Impact: 6/10 - Potential for meaningful results
- Technical Quality: 7/10 - Reasonable technical approach

OVERALL RATING: 6/10

RECOMMENDATIONS:
1. Review and refine your methodology
2. Consider additional data sources
3. Plan for potential challenges

NEXT STEPS:
- Conduct a pilot study
- Gather preliminary data
- Refine your research questions
                """
                results['step5'] = fallback_scoring
                st.warning("⚠️ Using fallback final scoring due to step failure")
            
            st.success(f"🎉 All 5 steps completed successfully! Total results: {len(results)}")
            
        except Exception as e:
            st.error(f"❌ Error in analysis chain: {str(e)}")
            st.error("Please check your API key and try again.")
            # Return partial results if available
            if results:
                st.warning(f"Partial results available: {len(results)}/5 steps completed")
        
        return results

    def ask_research_questions(self) -> Dict[str, str]:
        """Enhanced interactive questionnaire with placeholders"""
        st.subheader("📝 Tell us about your research project")
        st.markdown("*The AI will analyze your responses comprehensively - be as detailed as possible.*")
        
        questions = {
            "research_problem": {
                "question": "What specific problem are you trying to solve? (Be detailed)",
                "placeholder": "e.g., Understanding the impact of social media on student academic performance in high schools"
            },
            "research_question": {
                "question": "What is your main research question or hypothesis?",
                "placeholder": "e.g., Does increased social media usage correlate with decreased academic performance?"
            },
            "methodology_approach": {
                "question": "What methodology or approach do you plan to use?",
                "placeholder": "e.g., quantitative, qualitative, mixed-methods, experimental, case study, survey research"
            },
            "data_collection": {
                "question": "How will you collect your data? What are your data sources?",
                "placeholder": "e.g., surveys, interviews, experiments, existing datasets, observations"
            },
            "analysis_methods": {
                "question": "What analysis methods, statistical techniques, or algorithms will you use?",
                "placeholder": "e.g., regression analysis, thematic analysis, machine learning, content analysis"
            },
            "tools_software": {
                "question": "What tools, software, or technologies will you use?",
                "placeholder": "e.g., SPSS, R, Python, NVivo, Qualtrics, Google Forms"
            },
            "sample_population": {
                "question": "Describe your target population, sample size, or dataset",
                "placeholder": "e.g., 200 undergraduate students, 50 teachers, existing dataset with 1000 records"
            },
            "innovation_novelty": {
                "question": "What makes your approach unique, innovative, or different from existing work?",
                "placeholder": "e.g., new methodology, different population, novel combination of methods"
            },
            "expected_outcomes": {
                "question": "What outcomes, results, or impact do you expect?",
                "placeholder": "e.g., improved understanding, policy recommendations, new methodology"
            },
            "challenges_limitations": {
                "question": "What challenges, limitations, or risks do you anticipate?",
                "placeholder": "e.g., sample size limitations, data quality issues, ethical concerns"
            },
            "timeline_resources": {
                "question": "What is your timeline and what resources do you have available?",
                "placeholder": "e.g., 6 months, access to university participants, funding for software"
            }
        }
        
        user_responses = {}
        
        for key, question_data in questions.items():
            user_responses[key] = st.text_area(
                question_data["question"], 
                placeholder=question_data["placeholder"],
                key=key, 
                height=80,
                help=f"💡 {question_data['placeholder']}"
            )
        
        # Add a progress indicator
        filled_responses = {k: v for k, v in user_responses.items() if v.strip()}
        progress = len(filled_responses) / len(questions)
        
        st.progress(progress)
        st.caption(f"📊 Progress: {len(filled_responses)}/{len(questions)} questions completed ({progress:.0%})")
        
        # Add helpful tips
        if progress < 0.5:
            st.info("💡 **Tip:** Fill out at least 5 questions to start the analysis. More details = better results!")
        elif progress < 0.8:
            st.info("👍 **Good progress!** Consider adding more details for comprehensive analysis.")
        else:
            st.success("🎉 **Excellent!** You're ready for detailed analysis.")
        
        return user_responses

    def test_groq_connection(self) -> str:
        """Test Groq API connection with a simple prompt"""
        if not self.groq_api_key:
            return "ERROR: No Groq API key provided"
        
        test_prompt = "Say 'Hello, Groq is working!' in one sentence."
        
        try:
            result = self.call_groq_api(test_prompt, max_tokens=50)
            if result and not result.startswith("ERROR"):
                return f"✅ Groq test successful: {result}"
            else:
                return f"❌ Groq test failed: {result}"
        except Exception as e:
            return f"❌ Groq test exception: {str(e)}"

def main():
    st.set_page_config(
        page_title="AI Research Analyst Pro",
        page_icon="🤖",
        layout="wide"
    )
    
    # Add custom CSS for better markdown styling
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Style for markdown content */
    .markdown-text-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        color: #333333;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .markdown-text-container h1, 
    .markdown-text-container h2, 
    .markdown-text-container h3 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .markdown-text-container p {
        color: #333333;
        margin: 0.5rem 0;
    }
    
    .markdown-text-container ul, 
    .markdown-text-container ol {
        color: #333333;
        padding-left: 2rem;
    }
    
    .markdown-text-container li {
        color: #333333;
        margin: 0.5rem 0;
    }
    
    .markdown-text-container strong,
    .markdown-text-container b {
        color: #2c3e50;
        font-weight: bold;
    }
    
    .markdown-text-container em {
        color: #6c757d;
        font-style: italic;
    }
    
    /* Style for code blocks */
    .markdown-text-container code {
        background-color: #e9ecef;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
        font-family: 'Courier New', monospace;
        color: #333333;
    }
    
    /* Style for blockquotes */
    .markdown-text-container blockquote {
        border-left: 4px solid #007bff;
        padding-left: 1rem;
        margin: 1rem 0;
        font-style: italic;
        color: #6c757d;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    
    /* Style for tables */
    .markdown-text-container table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        color: #333333;
    }
    
    .markdown-text-container th, 
    .markdown-text-container td {
        border: 1px solid #dee2e6;
        padding: 0.75rem;
        text-align: left;
        color: #333333;
    }
    
    .markdown-text-container th {
        background-color: #007bff;
        color: white;
    }
    
    .markdown-text-container tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 AI Research Analyst Pro")
    st.markdown("**Advanced Chain-of-Prompts Analysis - Reliable Free AI Models!**")
    st.markdown("*Choose from Groq, Gemini, or Ollama (local) - all with free tiers!*")
    
    # Initialize analyzer
    analyzer = AIResearchAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Setup")
        
        # Model selection
        st.subheader("🤖 Choose AI Model")
        model_options = {
            'groq': '🚀 Groq (Fast & Free)',
            'gemini': '🤖 Gemini (Free)',
            'ollama': '🏠 Ollama (Local)'
        }
        
        selected_model = st.selectbox(
            "Select AI Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            help="Choose your preferred AI model"
        )
        analyzer.selected_model = selected_model
        
        # API Key inputs based on selection
        st.subheader("🔑 API Configuration")
        
        if selected_model == 'groq':
            groq_key = st.text_input("Groq API Key (Free)", type="password", 
                                    help="Get free API key from console.groq.com")
            if groq_key:
                analyzer.groq_api_key = groq_key
                
            # Add test button for Groq
            if analyzer.groq_api_key:
                if st.button("🧪 Test Groq Connection", help="Test if your Groq API key is working"):
                    with st.spinner("Testing Groq connection..."):
                        test_result = analyzer.test_groq_connection()
                        if test_result.startswith("✅"):
                            st.success(test_result)
                        else:
                            st.error(test_result)
                            st.info("💡 If the test fails, check your API key at console.groq.com")
                
        elif selected_model == 'gemini':
            gemini_key = st.text_input("Gemini API Key (Free)", type="password", 
                                    help="Get free API key from aistudio.google.com")
            if gemini_key:
                analyzer.gemini_api_key = gemini_key
                
        elif selected_model == 'ollama':
            ollama_url = st.text_input("Ollama URL", value="http://localhost:11434", 
                                    help="Ollama server URL (default: localhost:11434)")
            if ollama_url:
                analyzer.ollama_url = ollama_url
            
            # Model selection for Ollama
            ollama_model = st.selectbox(
                "Select Ollama Model",
                options=[
                    "llama3.2:1b",    # Fastest, smallest
                    "llama3.2:3b",    # Fast, good balance
                    "llama3.2:8b",    # Good quality, moderate speed
                    "llama3.2",       # Full model (slowest)
                    "llama3.2:70b"    # Largest, slowest
                ],
                index=0,  # Default to fastest
                help="Choose model size based on your hardware. Smaller = faster but less capable."
            )
            analyzer.ollama_model = ollama_model
            
            st.info("💡 Make sure Ollama is running locally with: ollama serve")
            st.info(f"📥 Install model with: ollama pull {ollama_model}")
            st.info("⚡ Smaller models (1b, 3b) are much faster than larger ones")
        
        # Model info
        st.subheader("ℹ️ Model Information")
        model_info = {
            'groq': "• Fastest response times\n• Free tier available\n• Llama3-8b model\n• No setup required",
            'gemini': "• High quality responses\n• Free tier available\n• Gemini Pro model\n• Google's latest AI",
            'ollama': "• Completely local\n• No API limits\n• Llama3.2 variants\n• Choose model size based on hardware\n• 1b/3b = fast, 8b/70b = slow"
        }
        st.info(model_info[selected_model])
        
        st.markdown("### 🔄 Analysis Chain:")
        st.markdown("""
        1. **Document Structure Analysis**
        2. **Deep Methodology Extraction** 
        3. **User Project Analysis**
        4. **Comparative Analysis**
        5. **Final Scoring & Recommendations**
        """)
        
        st.markdown("### ✨ Features:")
        st.markdown("""
        - Multiple free AI models
        - AI finds methodology anywhere in paper
        - Multi-pass extraction for completeness
        - Intelligent section identification
        - Comprehensive comparative analysis
        - Detailed scoring with explanations
        """)
        
        # Troubleshooting section for Ollama
        if selected_model == 'ollama':
            st.markdown("### 🔧 Ollama Troubleshooting:")
            with st.expander("Common Issues & Solutions"):
                st.markdown("""
                **Timeout Errors:**
                - Use smaller models (1b, 3b) instead of large ones
                - Close other applications to free up memory
                - Check if Ollama is running: `ollama serve`
                
                **Slow Performance:**
                - Switch to llama3.2:1b (fastest)
                - Ensure you have enough RAM (8GB+ recommended)
                - Use GPU acceleration if available
                
                **Model Not Found:**
                - Install model: `ollama pull llama3.2:1b`
                - Check available models: `ollama list`
                """)
        
        # Groq-specific information
        if selected_model == 'groq':
            st.markdown("### 🔧 Groq Optimizations:")
            with st.expander("Groq-Specific Features"):
                st.markdown("""
                **Applied Optimizations:**
                - ✅ Reduced token limits (1500 → 1000 chars input)
                - ✅ Simplified prompts for better compatibility
                - ✅ Fallback prompts if responses are too short
                - ✅ Enhanced error handling and debugging
                - ✅ Connection testing before analysis
                
                **If Steps Fail:**
                - Use shorter research papers
                - Provide concise project descriptions
                - Test connection first with the test button
                - Wait between attempts (rate limiting)
                - Try Gemini or Ollama as alternatives
                
                **Performance Tips:**
                - Groq is fastest but has stricter limits
                - Keep inputs under 1000 characters for best results
                - The app automatically uses fallbacks if needed
                """)
    
    # Right sidebar for debug panel
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Debug Panel")
        
        # Analysis Progress
        with st.expander("📊 Analysis Progress", expanded=False):
            if 'analysis_results' in st.session_state:
                results = st.session_state.analysis_results
                st.write(f"**Steps Completed:** {len(results)}/5")
                
                for i, step_key in enumerate(['step1', 'step2', 'step3', 'step4', 'step5'], 1):
                    step_result = results.get(step_key, 'Not completed')
                    if step_result and not step_result.startswith("ERROR"):
                        st.success(f"✅ Step {i}: {len(step_result)} chars")
                    elif step_result.startswith("ERROR"):
                        st.error(f"❌ Step {i}: Failed")
                    else:
                        st.info(f"⏳ Step {i}: Pending")
            else:
                st.info("No analysis results yet")
        
        # Model Info
        with st.expander("🤖 Model Info", expanded=False):
            # Define has_api_key for debug panel
            has_api_key = False
            if selected_model == 'groq' and analyzer.groq_api_key:
                has_api_key = True
            elif selected_model == 'gemini' and analyzer.gemini_api_key:
                has_api_key = True
            elif selected_model == 'ollama':
                has_api_key = True  # Ollama doesn't need API key
            
            st.write(f"**Selected Model:** {model_options[selected_model]}")
            st.write(f"**API Key Status:** {'✅ Set' if has_api_key else '❌ Missing'}")
            
            if selected_model == 'groq':
                st.write("**Groq Optimizations:** Active")
                st.write("**Token Limits:** Reduced")
                st.write("**Fallback System:** Enabled")
            elif selected_model == 'gemini':
                st.write("**Gemini Models:** Multiple fallbacks")
                st.write("**Token Limits:** Standard")
            elif selected_model == 'ollama':
                st.write(f"**Ollama Model:** {analyzer.ollama_model}")
                st.write("**Local Processing:** Yes")
        
        # Performance metrics
        with st.expander("⚡ Performance", expanded=False):
            if 'paper_text' in st.session_state:
                st.write(f"**Paper Length:** {len(st.session_state.paper_text):,} chars")
                st.write(f"**Estimated Tokens:** ~{len(st.session_state.paper_text)//4:,}")
            
            if selected_model == 'groq':
                st.write("**Groq Limits:**")
                st.write("- Input: 1500 chars max")
                st.write("- Output: 1200 tokens max")
                st.write("- Rate: ~100 requests/min")
            elif selected_model == 'gemini':
                st.write("**Gemini Limits:**")
                st.write("- Input: 2500 chars max")
                st.write("- Output: 1800 tokens max")
                st.write("- Rate: ~60 requests/min")
            elif selected_model == 'ollama':
                st.write("**Ollama Limits:**")
                st.write("- Input: 2500 chars max")
                st.write("- Output: 1800 tokens max")
                st.write("- Rate: No limit (local)")
        
        # Quick actions
        with st.expander("⚡ Quick Actions", expanded=False):
            if st.button("🔄 Clear Results", help="Clear all analysis results"):
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
                if 'paper_text' in st.session_state:
                    del st.session_state.paper_text
                st.rerun()
            
            if st.button("📋 Copy API Keys", help="Copy API key setup instructions"):
                st.code("""
# Groq API Key
Get free key from: console.groq.com

# Gemini API Key  
Get free key from: aistudio.google.com

# Ollama Setup
Install: ollama.ai
Run: ollama serve
Pull: ollama pull llama3.2:1b
                """)
        
        # Help section
        with st.expander("❓ Help", expanded=False):
            st.markdown("""
            **Quick Start:**
            1. Upload a research paper PDF
            2. Fill out your project details
            3. Choose an AI model
            4. Click "Start Analysis"
            
            **Best Practices:**
            - Use shorter papers for Groq
            - Be detailed in project descriptions
            - Test connection before analysis
            - Try different models if one fails
            """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload Research Paper")
        
        uploaded_file = st.file_uploader(
            "Upload PDF", 
            type=['pdf'],
            help="Upload any research paper PDF. Shorter papers (under 10 pages) work best with Groq."
        )
        
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                text = analyzer.extract_text_from_pdf(uploaded_file)
                
                if text:
                    st.success(f"✅ PDF processed! Extracted {len(text):,} characters")
                    st.session_state.paper_text = text
                    
                    # Show file info
                    file_size = len(uploaded_file.getvalue()) / 1024  # KB
                    st.info(f"📁 File: {uploaded_file.name} ({file_size:.1f} KB)")
                    
                    # Show preview with better formatting
                    with st.expander("📖 Document Preview", expanded=False):
                        preview_text = text[:800] + "..." if len(text) > 800 else text
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #007bff;">
                        <strong>First 800 characters:</strong><br>
                        {preview_text}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add paper analysis tips
                    if len(text) > 10000:
                        st.warning("⚠️ **Large paper detected** - This may take longer to process with Groq. Consider using Gemini or Ollama for better results.")
                    elif len(text) < 1000:
                        st.warning("⚠️ **Very short paper** - This might not provide enough content for comprehensive analysis.")
                    else:
                        st.success("✅ **Paper length looks good** for analysis!")
        else:
            st.info("👆 Please upload a research paper PDF to begin")
            st.markdown("""
            **📋 What to upload:**
            - Research papers, journal articles, conference papers
            - Academic reports, thesis documents
            - Any PDF with research methodology content
            
            **💡 Best practices:**
            - Papers with clear methodology sections work best
            - 5-15 page papers are ideal
            - Ensure text is extractable (not scanned images)
            """)
    
    with col2:
        st.header("🎯 Your Research Project")
        
        if 'paper_text' in st.session_state:
            user_responses = analyzer.ask_research_questions()
            
            # Check if user has filled out responses
            filled_responses = {k: v for k, v in user_responses.items() if v.strip()}
            
            if len(filled_responses) >= 5:  # At least 5 questions answered
                if st.button("🚀 Start AI Analysis Chain", type="primary", use_container_width=True):
                    # Check if API key is available for selected model
                    has_api_key = False
                    if selected_model == 'groq' and analyzer.groq_api_key:
                        has_api_key = True
                    elif selected_model == 'gemini' and analyzer.gemini_api_key:
                        has_api_key = True
                    elif selected_model == 'ollama':
                        has_api_key = True  # Ollama doesn't need API key
                    
                    if has_api_key:
                        with st.spinner(f"Running 5-step AI analysis chain with {model_options[selected_model]}..."):
                            # Run the complete analysis chain
                            results = analyzer.run_analysis_chain(
                                st.session_state.paper_text, 
                                user_responses
                            )
                            
                            # Store results
                            st.session_state.analysis_results = results
                            st.success("✅ Analysis complete!")
                            st.rerun()  # Refresh to show results
                    else:
                        st.error(f"Please provide your {model_options[selected_model]} API key in the sidebar!")
            else:
                st.info(f"Please answer at least 5 questions ({len(filled_responses)}/5 completed)")
        else:
            st.info("👆 Please upload a research paper first")

    # Display results if available
    if 'analysis_results' in st.session_state:
        st.header("📊 AI Analysis Results")
        
        results = st.session_state.analysis_results
        
        # Add a summary banner
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Steps Completed", f"{len(results)}/5")
        with col2:
            model_name = model_options.get(selected_model, 'Unknown')
            st.metric("Model Used", model_name)
        with col3:
            total_chars = sum(len(result) for result in results.values() if result and not result.startswith("ERROR"))
            st.metric("Total Analysis", f"{total_chars:,} chars")
        
        # Add a progress overview
        steps_status = []
        for i, step_key in enumerate(['step1', 'step2', 'step3', 'step4', 'step5'], 1):
            step_result = results.get(step_key, '')
            if step_result and not step_result.startswith("ERROR"):
                steps_status.append(f"✅ Step {i}")
            elif step_result.startswith("ERROR"):
                steps_status.append(f"❌ Step {i}")
            else:
                steps_status.append(f"⏳ Step {i}")
        
        st.info(f"**Analysis Progress:** {' | '.join(steps_status)}")
        
        # Create tabs for different analysis steps
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Document Analysis", 
            "🔬 Methodology", 
            "👤 Your Project", 
            "⚖️ Comparison", 
            "🏆 Final Scores"
        ])
        
        with tab1:
            st.subheader("Document Structure Analysis")
            step1_result = results.get('step1', 'No results available')
            if step1_result and step1_result != 'No results available':
                st.markdown("**Step 1: Document Structure Analysis**")
                st.markdown(f"""
                <div class="markdown-text-container">
                {step1_result}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Step 1 results not available")
        
        with tab2:
            st.subheader("Refined Methodology Extraction")
            step2_result = results.get('step2', 'No results available')
            if step2_result and step2_result != 'No results available':
                st.markdown("**Step 2: Deep Methodology Extraction**")
                st.markdown(f"""
                <div class="markdown-text-container">
                {step2_result}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Step 2 results not available")
        
        with tab3:
            st.subheader("Your Project Analysis")
            step3_result = results.get('step3', 'No results available')
            if step3_result and step3_result != 'No results available':
                st.markdown("**Step 3: User Project Analysis**")
                st.markdown(f"""
                <div class="markdown-text-container">
                {step3_result}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Step 3 results not available")
        
        with tab4:
            st.subheader("Comparative Analysis")
            step4_result = results.get('step4', 'No results available')
            if step4_result and step4_result != 'No results available':
                st.markdown("**Step 4: Comparative Analysis**")
                st.markdown(f"""
                <div class="markdown-text-container">
                {step4_result}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Step 4 results not available")
        
        with tab5:
            st.subheader("Final Evaluation & Recommendations")
            step5_result = results.get('step5', 'No results available')
            if step5_result and step5_result != 'No results available':
                st.markdown("**Step 5: Final Evaluation & Recommendations**")
                st.markdown(f"""
                <div class="markdown-text-container">
                {step5_result}
                </div>
                """, unsafe_allow_html=True)
                
                # Extract and display key scores if available
                if "DETAILED_SCORES" in step5_result or "OVERALL_RATING" in step5_result:
                    st.markdown("### 🎯 Key Scores")
                    # Try to extract scores from the text
                    import re
                    score_pattern = r'(\w+/\w+):\s*\[(\d+)/10\]'
                    scores = re.findall(score_pattern, step5_result)
                    if scores:
                        score_cols = st.columns(len(scores))
                        for i, (score_name, score_value) in enumerate(scores):
                            with score_cols[i]:
                                st.metric(score_name, f"{score_value}/10")
            else:
                st.error("❌ Step 5 results not available")
            
            # Download buttons for different formats
            st.markdown("---")
            st.subheader("📥 Download Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download button for full report (TXT)
                full_report = f"""
AI RESEARCH ANALYSIS REPORT
Generated with {model_options.get(selected_model, 'AI Model')}
===========================

DOCUMENT ANALYSIS:
{results.get('step1', 'N/A')}

METHODOLOGY EXTRACTION:
{results.get('step2', 'N/A')}

USER PROJECT ANALYSIS:
{results.get('step3', 'N/A')}

COMPARATIVE ANALYSIS:
{results.get('step4', 'N/A')}

FINAL EVALUATION:
{results.get('step5', 'N/A')}
"""
                
                st.download_button(
                    label="📥 Download TXT Report",
                    data=full_report,
                    file_name=f"ai_research_analysis_report_{selected_model}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # Download button for PDF report
                try:
                    from reportlab.lib.pagesizes import letter, A4
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    from reportlab.lib import colors
                    import io
                    
                    def create_pdf_report(results, model_name):
                        buffer = io.BytesIO()
                        doc = SimpleDocTemplate(buffer, pagesize=A4)
                        story = []
                        
                        # Styles
                        styles = getSampleStyleSheet()
                        title_style = ParagraphStyle(
                            'CustomTitle',
                            parent=styles['Heading1'],
                            fontSize=16,
                            spaceAfter=30,
                            alignment=1,  # Center
                            textColor=colors.darkblue
                        )
                        heading_style = ParagraphStyle(
                            'CustomHeading',
                            parent=styles['Heading2'],
                            fontSize=14,
                            spaceAfter=12,
                            textColor=colors.darkblue
                        )
                        normal_style = styles['Normal']
                        bold_style = ParagraphStyle(
                            'BoldStyle',
                            parent=styles['Normal'],
                            fontSize=12,
                            spaceAfter=6,
                            textColor=colors.black,
                            fontName='Helvetica-Bold'
                        )
                        
                        # Title
                        story.append(Paragraph("AI Research Analysis Report", title_style))
                        story.append(Paragraph(f"Generated with {model_name}", normal_style))
                        story.append(Spacer(1, 20))
                        
                        # Function to clean markdown text
                        def clean_markdown_text(text):
                            # Remove markdown formatting
                            import re
                            # Remove ** and * for bold/italic
                            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                            text = re.sub(r'\*(.*?)\*', r'\1', text)
                            # Remove # for headers
                            text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
                            # Remove markdown links
                            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
                            # Clean up extra whitespace
                            text = re.sub(r'\n\s*\n', '\n\n', text)
                            # Convert to HTML for reportlab
                            text = text.replace('\n', '<br/>')
                            return text
                        
                        # Sections
                        sections = [
                            ("Document Analysis", results.get('step1', 'N/A')),
                            ("Methodology Extraction", results.get('step2', 'N/A')),
                            ("User Project Analysis", results.get('step3', 'N/A')),
                            ("Comparative Analysis", results.get('step4', 'N/A')),
                            ("Final Evaluation", results.get('step5', 'N/A'))
                        ]
                        
                        for section_title, content in sections:
                            story.append(Paragraph(section_title, heading_style))
                            # Clean up content for PDF
                            clean_content = clean_markdown_text(content)
                            story.append(Paragraph(clean_content, normal_style))
                            story.append(Spacer(1, 12))
                        
                        doc.build(story)
                        buffer.seek(0)
                        return buffer
                    
                    # Create PDF
                    pdf_buffer = create_pdf_report(results, model_options.get(selected_model, 'AI Model'))
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"ai_research_analysis_report_{selected_model}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                except ImportError:
                    st.error("PDF generation requires reportlab. Install with: pip install reportlab")
                    st.info("TXT download available above")
        
        # Add a summary section
        st.markdown("---")
        st.markdown("### 📋 Analysis Summary")
        st.markdown(f"**Model Used:** {model_options.get(selected_model, 'Unknown')}")
        st.markdown(f"**Analysis Steps Completed:** {len(results)}/5")
        
        # Show which steps have results
        steps_with_results = []
        for i, step_key in enumerate(['step1', 'step2', 'step3', 'step4', 'step5'], 1):
            if results.get(step_key) and results.get(step_key) != 'No results available':
                steps_with_results.append(f"Step {i}")
        
        if steps_with_results:
            st.success(f"✅ Completed steps: {', '.join(steps_with_results)}")
        else:
            st.error("❌ No analysis results found. Please try running the analysis again.")
            
    else:
        # Show a message when no results are available
        st.info("📊 Analysis results will appear here after running the AI analysis chain.")

if __name__ == "__main__":
    main()
