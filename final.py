import os
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
#
# Get the port from the environment (Cloud Run provides PORT=8080)
PORT = int(os.getenv("PORT", 8080))

# Load environment variables from .env file
load_dotenv()

# Access the API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set API Key in os,
# If fails, un-comment it. Commented for GCP cloud run
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

# Initialize LLM and Search API
llm = ChatGroq(temperature=0.7, model_name="groq/llama3-8b-8192", api_key=GROQ_API_KEY)
tavily = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)

# Initialize session state for storing results
if "industry_results" not in st.session_state:
    st.session_state.industry_results = None
if "use_case_results" not in st.session_state:
    st.session_state.use_case_results = None
if "dataset_results" not in st.session_state:
    st.session_state.dataset_results = None

# Function to fetch company information
def get_company_info(company_name):
    query = f"{company_name} company overview, industry segment, key offerings, strategic focus"
    results = tavily.results(query)[:3]
    return results

# Function to generate a PDF
def generate_pdf(content, filename):
    """Generate a PDF from the given content."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='', size=12)

    if not isinstance(content, str):  # Convert CrewOutput to string
        content = str(content)

    for line in content.split("\n"):
        pdf.cell(0, 10, line, ln=True)

    pdf.output(filename)

# Streamlit UI
st.title("AI-Powered Industry Research & Use Case Generator")

# User input for company name
company_name = st.text_input("Enter the company name:")

if st.button("Generate Report"):
    with st.spinner("Fetching company information..."):
        company_info = get_company_info(company_name)

    st.markdown("### **Company Information**")
    for i, res in enumerate(company_info, 1):
        st.markdown(f"**Result {i}:** {res}")

    # Industry Research Agent
    industry_researcher = Agent(
        llm=llm,
        role="Industry Research Specialist",
        goal="Analyze company details and generate industry insights.",
        backstory="Expert in industry analysis, financial trends, and market predictions.",
    )

    industry_task = Task(
        description=f"Analyze company details:\n\n{company_info}\n\nExtract industry segment, key offerings, and strategic focus areas.",
        agent=industry_researcher,
        expected_output="A structured summary of the company's industry, key offerings, and strategic focus areas."
    )

    # Use Case Generator Agent
    use_case_generator = Agent(
        llm=llm,
        role="AI/ML Use Case Analyst",
        goal="Identify AI/ML/GenAI use cases for the company.",
        backstory="Specializes in AI applications across industries.",
    )

    use_case_task = Task(
        description="Generate 5 AI/ML/GenAI use cases relevant to the company.",
        agent=use_case_generator,
        expected_output="A list of at least 5 structured AI/ML/GenAI use cases for the company."
    )

    # Resource Collector Agent
    resource_collector = Agent(
        llm=llm,
        role="AI Resource Researcher",
        goal="Find relevant datasets for AI/ML applications.",
        backstory="Data scientist specializing in dataset discovery.",
    )

    resource_task = Task(
        description="Find relevant public datasets for AI/ML use cases.",
        agent=resource_collector,
        expected_output="A dictionary where keys are AI/ML applications and values are dataset links."
    )

    # Execute Agents
    crew1 = Crew(agents=[industry_researcher], tasks=[industry_task])
    crew2 = Crew(agents=[use_case_generator], tasks=[use_case_task])
    crew3 = Crew(agents=[resource_collector], tasks=[resource_task])

    with st.spinner("Generating Industry Research Report..."):
        st.session_state.industry_results = str(crew1.kickoff())

    with st.spinner("Generating AI Use Cases..."):
        st.session_state.use_case_results = str(crew2.kickoff())

    with st.spinner("Finding AI/ML Datasets..."):
        st.session_state.dataset_results = str(crew3.kickoff())

# Display Results
if st.session_state.industry_results:
    st.markdown("## **Industry Research Report**")
    st.markdown(f"```\n{st.session_state.industry_results}\n```")

if st.session_state.use_case_results:
    st.markdown("## **AI Use Cases**")
    st.markdown(f"```\n{st.session_state.use_case_results}\n```")

if st.session_state.dataset_results:
    st.markdown("## **AI/ML Datasets**")
    st.markdown(f"```\n{st.session_state.dataset_results}\n```")

# Save results to PDFs
industry_pdf = "industry_report.pdf"
use_case_pdf = "use_cases.pdf"
dataset_pdf = "datasets.pdf"

if st.session_state.industry_results:
    generate_pdf(st.session_state.industry_results, industry_pdf)

if st.session_state.use_case_results:
    generate_pdf(st.session_state.use_case_results, use_case_pdf)

if st.session_state.dataset_results:
    generate_pdf(st.session_state.dataset_results, dataset_pdf)

# Provide download links
allfile=False #will Truw when download is ready


if st.session_state.industry_results:
    st.sidebar.markdown("## **Download Reports**")
    st.sidebar.download_button("Download Industry Report", open(industry_pdf, "rb"), industry_pdf, "application/pdf")

if st.session_state.use_case_results:
    st.sidebar.download_button("Download AI Use Cases", open(use_case_pdf, "rb"), use_case_pdf, "application/pdf")

if st.session_state.dataset_results:
    st.sidebar.download_button("Download AI/ML Datasets", open(dataset_pdf, "rb"), dataset_pdf, "application/pdf")
    allfile=True

# Save all data to a text file
all_results_text = f"""
=== Industry Research Report ===
{st.session_state.industry_results}

=== AI Use Cases ===
{st.session_state.use_case_results}

=== AI/ML Datasets ===
{st.session_state.dataset_results}
"""

text_filename = "full_report.txt"
with open(text_filename, "w", encoding="utf-8") as f:
    f.write(all_results_text)

if allfile:
# Provide text file download
    st.sidebar.download_button("Download Knowledge Base (Text)", open(text_filename, "rb"), text_filename, "text/plain")


# 🔹 **FAISS + TF-IDF for Retrieval**
vectorizer = TfidfVectorizer()
corpus = all_results_text.split("\n\n")  # Split by paragraphs

# Convert text to TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(corpus)
index = faiss.IndexFlatL2(tfidf_matrix.shape[1])
index.add(tfidf_matrix.toarray())  # Convert sparse matrix to dense

# Store FAISS index
with open("faiss_index.pkl", "wb") as f:
    pickle.dump(index, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

if allfile:
    # 🔹 **Sidebar Chatbot Integration**
    st.sidebar.header("📢 Chatbot Assistant")
    st.sidebar.write("Ask questions related to the generated reports!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages in sidebar
    for msg in st.session_state.messages:
        with st.sidebar.chat_message(msg["role"]):
            st.sidebar.write(msg["content"])

    # Get user input in sidebar
    if user_input := st.sidebar.chat_input("Ask something about the reports..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 🔹 **Retrieve Most Similar Content from FAISS**
        query_vector = vectorizer.transform([user_input]).toarray()
        _, idx = index.search(np.array(query_vector), k=1)  # Get closest match index
        retrieved_text = corpus[idx[0][0]]

        # 🔹 **Pass Retrieved Text to Groq for Answering**
        llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192", api_key=GROQ_API_KEY)

        response = llm.invoke(
            f"You are a helpful assistant. Answer in short and casually. dont answer other than context Only respond based on the following retrieved knowledge:\n\n"
            f"{retrieved_text}\n\nUser query: {user_input}"
        )

        st.session_state.messages.append({"role": "assistant", "content": response.content})

        with st.sidebar.chat_message("assistant"):
            st.sidebar.write(response.content)

# Run Streamlit on the correct port
# if __name__ == "__main__":
#     st.run(host="0.0.0.0", port=PORT)
