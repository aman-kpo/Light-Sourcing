import streamlit as st
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import os
import json

# Load environment variables
# load_dotenv()

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialize the ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.81,api_key = GOOGLE_API_KEY)

class Fit_score(BaseModel):
    full_name: str = Field(description="The name of the candidate")
    linkedin_url: str = Field(description="The linkedin url of the candidate")
    Current_title_comp: str = Field(description="The current company and role the candidate is working")
    degree: str = Field(description="The education background of the candidate")
    yrs_of_experience: str = Field(description="The duration of the candidate as a working professional")
    tech_stack: str = Field(description="Key Tech Stack the candidate is proficient in")
    fit_score: float = Field(description="Fit Score (1-10) indicating how good the user is for the job")
    key_heighlights: str = Field(description="Major milestones achieved by the user")
    Message_templates: List[str] = Field(description="Some sample message templates for the user asking about the interest in this role.")

sum_llm = llm.with_structured_output(Fit_score)

st.set_page_config(page_title="SRN Sourcing Agent", layout="wide")

st.title("SRN AI Sourcing Agent")

# Create a session state to store the uploaded CSV file
if 'df' not in st.session_state:
    st.session_state.df = None

if 'job_details' not in st.session_state:
    st.session_state.job_details = None

# Upload CSV file
st.header("Upload CSV")
st.write("Upload a CSV file containing LinkedIn profile URLs")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df

    # Show dataframe preview
    st.subheader("CSV Preview")
    st.dataframe(df.head())

if st.session_state.df is not None:
    # Enter job details
    st.header("Enter Job Details")
    job_title = st.text_input("Job Title", placeholder="Senior Software Engineer")
    tech_stack = st.text_input("Key Tech Stack (comma-separated)", placeholder="Python, AWS, React")
    location = st.text_input("Location", placeholder="Remote / San Francisco / New York")
    domain = st.text_input("Domain (Optional)", placeholder="FinTech, AI, SaaS")
    salary = st.text_input("Salary (Optional)", placeholder="$150k - $200k")
    

    if st.button("Start Sourcing!"):
        st.session_state.job_details = {
            'job_title': job_title,
            'tech_stack': tech_stack,
            'location': location,
            'domain': domain,
            
       'salary': salary
        }

        SYSTEM_PROMPT = """
        You are an elite AI recruiter for SRN specializing in sourcing software engineers for VC-backed startups.
        Strictly apply these REJECTION RULES:
         No CS degree (or equivalent)
         No VC-backed startup experience (Seed–Series C/D)
         Only FAANG/Big Tech experience
         Less than 3 years post-grad work
         More than one job <2 years tenure
         Enterprise/Consulting background (Infosys, Wipro, Cognizant, Dell, Cisco, etc)
         Requires visa sponsorship (H1B, OPT, TN)
        For candidates who pass:
         Calculate Fit Score (1–10) based on:
        - Problem Solving (20%)
        - 0→1 Building (20%)
        - Startup Experience (20%)
        - Tech Stack Alignment (15%)
        - Tenure & Stability (15%)
        - Domain Relevance (10%)
         Only shortlist candidates with Fit Score ≥ 9.2
         Generate 4 Connect Templates under 300 characters (2 with salary, 2 without)
        """

        query_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """
        You are an expert Recruitor, your task is to determine based on user profile summary  and the job details, 
             if the candidate is a good fit for the job. Make sure to extract only relevent info from the following inputs for assessing candidate:
             Candidate details:
        candidate_summary:{summary}
        candidate_name:{name}
        linkedin_url:{url}
        Job details:
             
        Job_title: {Job_title}
        tech_stack:{tech_stack}
        location:{location}
        domain:{domain} 
        salary:{salary}
        """),
        ])

        cat_class = query_prompt | sum_llm

        df_score = pd.DataFrame()
        names = []
        urls = []
        curr = []
        degree = []
        yrs = []
        tech = []
        score = []
        key = []
        message_temp1 = []
        message_temp2 = []
        message_temp3 = []
        message_temp4 = []

        for i in range(len(st.session_state.df)):
            name = str(df["full_name"].values[i])
            url = str(df["URL"].values[i])
            summary = str(df["Summary"].values[i])

            op = cat_class.invoke({
                "summary": summary,
                "name": name,
                "url": url,
                "Job_title": job_title,
                "tech_stack": tech_stack,
                "location": location,
                "domain": domain,
                "salary": salary
            })

            names.append(op.full_name)
            urls.append(op.linkedin_url)
            curr.append(op.Current_title_comp)
            degree.append(op.degree)
            yrs.append(op.yrs_of_experience)
            tech.append(op.tech_stack)
            score.append(op.fit_score)
            key.append(op.key_heighlights)

            message_temp1.append(f"Hi {op.full_name}, I came across your profile and was impressed with your experience in {op.tech_stack}.")
            message_temp2.append(f"Hi {op.full_name}, I'm reaching out from {domain} and we're looking for a {job_title} with expertise in {op.tech_stack}.")
            message_temp3.append(f"Hi {op.full_name}, I saw your profile and was impressed with your achievements in {op.key_heighlights}. We're looking for a {job_title} with similar skills and experience. Salary range is ${salary}.")
            message_temp4.append(f"Hi {op.full_name}, I'm excited to share an opportunity for a {job_title} role at {domain} that aligns with your skills in {op.tech_stack}. If you're interested, let's schedule a call to discuss further.")

            time.sleep(5)

        df_score["name"] = names
        df_score["url"] = urls
        df_score["Current Title"] = curr
        df_score["Degree"] = degree
        df_score["Years of experience"] = yrs
        df_score["Tech Stack"] = tech
        df_score["Fit Score"] = score
        df_score["Key Highlights"] = key
        df_score["Message_Template1"] = message_temp1
        df_score["Message_Template2"] = message_temp2
        df_score["Message_Template3"] = message_temp3
        df_score["Message_Template4"] = message_temp4

        df_score = df_score.sort_values(by="Fit Score", ascending=False)

        st.header("Shortlisted Candidates")
        st.dataframe(df_score.head(10))

        @st.cache
        def convert_df(df):
            return df.to_csv(index=False)

        csv = convert_df(df_score)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='shortlisted_candidates.csv',
            mime='text/csv',
        )
