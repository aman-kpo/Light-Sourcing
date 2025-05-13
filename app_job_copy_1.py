import streamlit as st
import pandas as pd
import json
import os
from pydantic import BaseModel, Field
from typing import List, Set, Dict, Any, Optional
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import gspread
from google.oauth2 import service_account

st.set_page_config(
    page_title="Candidate Matching App",
    page_icon="👨‍💻🎯",
    layout="wide"
)

# Define pydantic model for structured output
class Shortlist(BaseModel):
    fit_score: float = Field(description="A score between 0 and 10 indicating how closely the candidate profile matches the job requirements.")
    candidate_name: str = Field(description="The name of the candidate.")
    candidate_url: str = Field(description="The URL of the candidate's LinkedIn profile.")
    candidate_summary: str = Field(description="A brief summary of the candidate's skills and experience along with its educational background.")
    candidate_location: str = Field(description="The location of the candidate.")
    justification: str = Field(description="Justification for the shortlisted candidate with the fit score")

# Function to parse and normalize tech stacks
def parse_tech_stack(stack):
    if pd.isna(stack) or stack == "" or stack is None:
        return set()
    if isinstance(stack, set):
        return stack
    try:
        # Handle potential string representation of sets
        if isinstance(stack, str) and stack.startswith("{") and stack.endswith("}"):
            # This could be a string representation of a set
            items = stack.strip("{}").split(",")
            return set(item.strip().strip("'\"") for item in items if item.strip())
        return set(map(lambda x: x.strip().lower(), str(stack).split(',')))
    except Exception as e:
        st.error(f"Error parsing tech stack: {e}")
        return set()

def display_tech_stack(stack_set):
    if isinstance(stack_set, set):
        return ", ".join(sorted(stack_set))
    return str(stack_set)

def get_matching_candidates(job_stack, candidates_df):
    """Find candidates with matching tech stack for a specific job"""
    matched = []
    job_stack_set = parse_tech_stack(job_stack)
    
    for _, candidate in candidates_df.iterrows():
        candidate_stack = parse_tech_stack(candidate['Key Tech Stack'])
        common = job_stack_set & candidate_stack
        if len(common) >= 2:
            matched.append({
                "Name": candidate["Full Name"],
                "URL": candidate["LinkedIn URL"],
                "Degree & Education": candidate["Degree & University"],
                "Years of Experience": candidate["Years of Experience"],
                "Current Title & Company": candidate['Current Title & Company'],
                "Key Highlights": candidate["Key Highlights"],
                "Location": candidate["Location (from most recent experience)"],
                "Tech Stack": candidate_stack
            })
    return matched

def setup_llm():
    """Set up the LangChain LLM with structured output"""
    # Create LLM instance
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Create structured output
    sum_llm = llm.with_structured_output(Shortlist)
    
    # Create system prompt
    system = """You are an expert Recruitor, your task is to analyse the Candidate profile and determine if it matches with the job details and provide a score(out of 10) indicating how compatible the
    the profile is according to job.
Try to ensure following points while estimating the candidate's fit score:
For education:
Tier1 - MIT, Stanford, CMU, UC Berkeley, Caltech, Harvard, IIT Bombay, IIT Delhi, Princeton, UIUC, University of Washington, Columbia, University of Chicago, Cornell, University of Michigan (Ann Arbor), UT Austin - Maximum points
Tier2 - UC Davis, Georgia Tech, Purdue, UMass Amherst,etc - Moderate points
Tier3 - Unknown or unranked institutions - Lower points or reject

Startup Experience Requirement:
Candidates must have worked  as a direct employee at a VC-backed startup (Seed to series C/D)
preferred - Y Combinator, Sequoia,a16z,Accel,Founders Fund,LightSpeed,Greylock,Benchmark,Index Ventures,etc.   

    The fit score signifies based on following metrics:
    1–5 - Poor Fit - Auto-reject
    6–7 - Weak Fit - Auto-reject
    8.0–8.7 - Moderate Fit - Auto-reject
    8.8–10 - STRONG Fit - Include in results
    Also look if the candidate haven't on a similar role in the past reject it directly
    """
    
    # Create query prompt
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", """
    You are an expert Recruitor, your task is to determine if the user is a correct match for the given job or not.
          For this you will be provided with the follwing inputs of job and candidates:
    Job Details
    Company: {Company}
    Role: {Role}
    About Company: {desc}
    Locations: {Locations}
    Tech Stack: {Tech_Stack}
    Industry: {Industry}

    Candidate Details:
    Full Name: {Full_Name}
    LinkedIn URL: {LinkedIn_URL}
    Current Title & Company: {Current_Title_Company}
    Years of Experience: {Years_of_Experience}
    Degree & University: {Degree_University}
    Key Tech Stack: {Key_Tech_Stack}
    Key Highlights: {Key_Highlights}
    Location (from most recent experience): {cand_Location}
    Past_Experience: {Experience}


    Answer in the structured manner as per the schema.
    If any parameter is Unknown try not to include in the summary, only include those parameters which are known.
    Also do not use Unknown parameters in the evaluating candidates and estimating fit scores.
    """),
    ])
    
    # Chain the prompt and LLM
    cat_class = query_prompt | sum_llm
    
    return cat_class

def call_llm(candidate_data, job_data, llm_chain):
    """Call the actual LLM to evaluate the candidate"""
    try:
        # Convert tech stacks to strings for the LLM payload
        job_tech_stack = job_data.get("Tech_Stack", set())
        candidate_tech_stack = candidate_data.get("Tech Stack", set())
        
        if isinstance(job_tech_stack, set):
            job_tech_stack = ", ".join(sorted(job_tech_stack))
            
        if isinstance(candidate_tech_stack, set):
            candidate_tech_stack = ", ".join(sorted(candidate_tech_stack))
        
        # Prepare payload for LLM
        payload = {
            "Company": job_data.get("Company", ""),
            "Role": job_data.get("Role", ""),
            "desc": job_data.get("desc", ""),
            "Locations": job_data.get("Locations", ""),
            "Tech_Stack": job_tech_stack,
            "Industry": job_data.get("Industry", ""),

            "Full_Name": candidate_data.get("Name", ""),
            "LinkedIn_URL": candidate_data.get("URL", ""),
            "Current_Title_Company": candidate_data.get("Current Title & Company", ""),
            "Years_of_Experience": candidate_data.get("Years of Experience", ""),
            "Degree_University": candidate_data.get("Degree & Education", ""),
            "Key_Tech_Stack": candidate_tech_stack,
            "Key_Highlights": candidate_data.get("Key Highlights", ""),
            "cand_Location": candidate_data.get("Location", ""),
            "Experience": candidate_data.get("Experience", "")
        }
        
        # Call LLM
        response = llm_chain.invoke(payload)
        
        # Return response in expected format
        return {
            "candidate_name": response.candidate_name,
            "candidate_url": response.candidate_url,
            "candidate_summary": response.candidate_summary,
            "candidate_location": response.candidate_location,
            "fit_score": response.fit_score,
            "justification": response.justification
        }
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        # Fallback to a default response
        return {
            "candidate_name": candidate_data.get("Name", "Unknown"),
            "candidate_url": candidate_data.get("URL", ""),
            "candidate_summary": "Error processing candidate profile",
            "candidate_location": candidate_data.get("Location", "Unknown"),
            "fit_score": 0.0,
            "justification": f"Error in LLM processing: {str(e)}"
        }

def process_candidates_for_job(job_row, candidates_df, llm_chain=None):
    """Process candidates for a specific job using the LLM"""
    if llm_chain is None:
        with st.spinner("Setting up LLM..."):
            llm_chain = setup_llm()
    
    selected_candidates = []
    
    try:
        # Get job-specific data
        job_data = {
            "Company": job_row["Company"],
            "Role": job_row["Role"],
            "desc": job_row.get("One liner", ""),
            "Locations": job_row.get("Locations", ""),
            "Tech_Stack": job_row["Tech Stack"],
            "Industry": job_row.get("Industry", "")
        }
        
        # Find matching candidates for this job
        with st.spinner("Finding matching candidates based on tech stack..."):
            matching_candidates = get_matching_candidates(job_row["Tech Stack"], candidates_df)
        
        if not matching_candidates:
            st.warning("No candidates with matching tech stack found for this job.")
            return []
        
        st.success(f"Found {len(matching_candidates)} candidates with matching tech stack.")
        
        # Create progress elements
        candidates_progress = st.progress(0)
        candidate_status = st.empty()
        
        # Process each candidate
        for i, candidate_data in enumerate(matching_candidates):
            # Update progress
            candidates_progress.progress((i + 1) / len(matching_candidates))
            candidate_status.text(f"Evaluating candidate {i+1}/{len(matching_candidates)}: {candidate_data.get('Name', 'Unknown')}")
            
            # Process the candidate with the LLM
            response = call_llm(candidate_data, job_data, llm_chain)
            
            response_dict = {
                "Name": response["candidate_name"],
                "LinkedIn": response["candidate_url"],
                "summary": response["candidate_summary"],
                "Location": response["candidate_location"],
                "Fit Score": response["fit_score"],
                "justification": response["justification"],
                # Add back original candidate data for context
                "Educational Background": candidate_data.get("Degree & Education", ""),
                "Years of Experience": candidate_data.get("Years of Experience", ""),
                "Current Title & Company": candidate_data.get("Current Title & Company", "")
            }
            
            # Add to selected candidates if score is high enough
            if response["fit_score"] >= 8.8:
                selected_candidates.append(response_dict)
                st.markdown(response_dict)
            else:
                st.write(f"Rejected candidate: {response_dict['Name']} with score: {response['fit_score']}")
        
        # Clear progress indicators
        candidates_progress.empty()
        candidate_status.empty()
        
        # Show results
        if selected_candidates:
            st.success(f"✅ Found {len(selected_candidates)} suitable candidates for this job!")
        else:
            st.info("No candidates met the minimum fit score threshold for this job.")
        
        return selected_candidates
        
    except Exception as e:
        st.error(f"Error processing job: {e}")
        return []

def main():
    st.title("👨‍💻 Candidate Matching App")
    secret_content = st.secrets["GCP_SERVICE_ACCOUNT"]
    secret_content = secret_content.replace("\n", "\\n")
    secret_content = json.loads(secret_content)
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    # Initialize session state
    if 'processed_jobs' not in st.session_state:
        st.session_state.processed_jobs = {}
    
    st.write("""
    This app matches job listings with candidate profiles based on tech stack and other criteria.
    Select a job to find matching candidates.
    """)
    
    # API Key input
    with st.sidebar:
        st.header("API Configuration")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API Key set!")
        else:
            st.warning("Please enter OpenAI API Key to use LLM features")
            
    # Show API key warning if not set
    creds = service_account.Credentials.from_service_account_info(secret_content, scopes=SCOPES)
    gc = gspread.authorize(creds)
    job_sheet = gc.open_by_key('1BZlvbtFyiQ9Pgr_lpepDJua1ZeVEqrCLjssNd6OiG9k')
    candidates_sheet = gc.open_by_key('1u_9o5f0MPHFUSScjEcnA8Lojm4Y9m9LuWhvjYm6ytF4')

    if not api_key:
        st.warning("⚠️ You need to provide an OpenAI API key in the sidebar to use this app.")

    if api_key:
        try:
            # Load data from Google Sheets
            job_worksheet = job_sheet.worksheet('paraform_jobs_formatted')
            job_data = job_worksheet.get_all_values()
            candidate_worksheet = candidates_sheet.worksheet('transformed_candidates_updated')
            candidate_data = candidate_worksheet.get_all_values()
            
            # Convert to DataFrames
            jobs_df = pd.DataFrame(job_data[1:], columns=job_data[0])
            candidates_df = pd.DataFrame(candidate_data[1:], columns=candidate_data[0])
            
            # Display data preview
            with st.expander("Preview uploaded data"):
                st.subheader("Jobs Data Preview")
                st.dataframe(jobs_df.head(3))
                
                st.subheader("Candidates Data Preview")
                st.dataframe(candidates_df.head(3))
            
            # Map column names if needed
            column_mapping = {
                "Full Name": "Full Name",
                "LinkedIn URL": "LinkedIn URL",
                "Current Title & Company": "Current Title & Company",
                "Years of Experience": "Years of Experience",
                "Degree & University": "Degree & University",
                "Key Tech Stack": "Key Tech Stack", 
                "Key Highlights": "Key Highlights",
                "Location (from most recent experience)": "Location (from most recent experience)"
            }
            
            # Rename columns if they don't match expected
            candidates_df = candidates_df.rename(columns={
                col: mapping for col, mapping in column_mapping.items() 
                if col in candidates_df.columns and col != mapping
            })
            
            # Now, instead of processing all jobs upfront, we'll display job selection
            # and only process the selected job when the user chooses it
            display_job_selection(jobs_df, candidates_df)
            
        except Exception as e:
            st.error(f"Error processing files: {e}")
    
    st.divider()
    

def display_job_selection(jobs_df, candidates_df):
    # Store the LLM chain as a session state to avoid recreating it
    if 'llm_chain' not in st.session_state:
        st.session_state.llm_chain = None
    
    st.subheader("Select a job to view potential matches")
    
    # Create job options - but don't compute matches yet
    job_options = []
    for i, row in jobs_df.iterrows():
        job_options.append(f"{row['Role']} at {row['Company']}")
    
    if job_options:
        selected_job_index = st.selectbox("Jobs:", 
                                      range(len(job_options)),
                                      format_func=lambda x: job_options[x])
        
        # Display job details
        job_row = jobs_df.iloc[selected_job_index]
        
        # Parse tech stack for display
        job_row_stack = parse_tech_stack(job_row["Tech Stack"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Job Details: {job_row['Role']}")
            
            job_details = {
                "Company": job_row["Company"],
                "Role": job_row["Role"],
                "Description": job_row.get("One liner", "N/A"),
                "Locations": job_row.get("Locations", "N/A"),
                "Industry": job_row.get("Industry", "N/A"),
                "Tech Stack": display_tech_stack(job_row_stack)
            }
            
            for key, value in job_details.items():
                st.markdown(f"**{key}:** {value}")
        
        # Create a key for this job in session state
        job_key = f"job_{selected_job_index}_processed"
        
        if job_key not in st.session_state:
            st.session_state[job_key] = False
        
        # Add a process button for this job
        if not st.session_state[job_key]:
            if st.button(f"Find Matching Candidates for this Job"):
                if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
                    st.error("Please enter your OpenAI API key in the sidebar before processing")
                else:
                    # Process candidates for this job (only when requested)
                    selected_candidates = process_candidates_for_job(
                        job_row, 
                        candidates_df,
                        st.session_state.llm_chain
                    )
                    
                    # Store the results and set as processed
                    if 'Selected_Candidates' not in st.session_state:
                        st.session_state.Selected_Candidates = {}
                    st.session_state.Selected_Candidates[selected_job_index] = selected_candidates
                    st.session_state[job_key] = True
                    
                    # Store the LLM chain for reuse
                    if st.session_state.llm_chain is None:
                        st.session_state.llm_chain = setup_llm()
                    
                    # Force refresh
                    st.rerun()
        
        # Display selected candidates if already processed
        if st.session_state[job_key] and 'Selected_Candidates' in st.session_state:
            selected_candidates = st.session_state.Selected_Candidates.get(selected_job_index, [])
            
            # Display selected candidates
            st.subheader("Selected Candidates")
            
            if len(selected_candidates) > 0:
                for i, candidate in enumerate(selected_candidates):
                    with st.expander(f"{i+1}. {candidate['Name']} (Score: {candidate['Fit Score']})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Summary:** {candidate['summary']}")
                            st.markdown(f"**Current:** {candidate['Current Title & Company']}")
                            st.markdown(f"**Education:** {candidate['Educational Background']}")
                            st.markdown(f"**Experience:** {candidate['Years of Experience']}")
                            st.markdown(f"**Location:** {candidate['Location']}")
                            st.markdown(f"**[LinkedIn Profile]({candidate['LinkedIn']})**")
                        
                        with col2:
                            st.markdown(f"**Fit Score:** {candidate['Fit Score']}")
                        
                        st.markdown("**Justification:**")
                        st.info(candidate['justification'])
            else:
                st.info("No candidates met the minimum score threshold (8.8) for this job.")
                
                # We don't show tech-matched candidates here since they are generated
                # during the LLM matching process now
            
            # Add a reset button to start over
            if st.button("Reset and Process Again"):
                st.session_state[job_key] = False
                st.rerun()

if __name__ == "__main__":
    main()
