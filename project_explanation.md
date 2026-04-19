# Intelligent Credit Risk Scoring & Agentic Lending Decision Support System

## 1. Problem Understanding
Deciding who receives a loan (Credit Risk Scoring) is traditionally a complex financial challenge. Mistakes can lead to significant financial loss for a bank or result in unfair rejections for individuals. In Milestone 1, we tackled this by building a Machine Learning (ML) model using historical borrower data. The model provided a statistical "Probability of Default." 

However, purely algorithmic decisions are often "black boxes," leaving humans without understandable justifications. In Milestone 2, our goal was to transform this simple ML system into an **Agentic AI Lending Assistant**. This system doesn't immediately accept or reject; instead, it looks at the ML output, searches through real ethical guidelines, and generates a structured, explainable report. 

## 2. System Workflow (How Data Flows)
1. **User Input:** A human (e.g., loan officer) enters the borrower's details (age, income, late payments, etc.) into a simple User Interface (UI).
2. **Machine Learning Predicts:** The existing trained model (Decision Tree) looks at the profile and predicts a purely mathematical risk score.
3. **Agent is Triggered:** Instead of stopping, our AI Agent steps in. It takes the borrower's data and the ML risk score.
4. **Retrieval (RAG):** The Agent searches a local "Knowledge Base" (database of text files) containing lending policies and ethics rules that match the situation.
5. **Report Generation (LLM):** The Agent passes the collected data (borrower profile + ML Score + retrieved rules) to a Large Language Model (Llama 3). The LLM processes everything to output a structured lending report.
6. **UI Display & PDF Export:** The generated report is displayed back on the screen to the loan officer, who can act on it and download it as a convenient PDF file.

## 3. Why ML Model Chosen
In Milestone 1, both Logistic Regression and Decision Tree were trained. We chose the **Decision Tree**. Decision Trees are slightly more intuitive than complex mathematical equations. They split decisions based on distinct thresholds (e.g., "If debt ratio > 0.5 then..."), making them easier to map to real-world financial logic.

## 4. Why an LLM is Used
A standard ML model just outputs a number (e.g., 85% risk). A Large Language Model (LLM) understands language and context. We use it to explain *why* that 85% risk matters, contextualizing it with real human language and providing distinct, structured advice rather than just a red error metric.

## 5. Why Retrieval-Augmented Generation (RAG)
LLMs are powerful but prone to "hallucinating" (inventing things) if they don't know the exact laws of a specific bank or country. RAG solves this. Before the LLM thinks about the loan decision, we forcefully inject our custom rulebook (the *knowledge_base*) into its prompt context. This limits the AI, making it rely exclusively on our approved ethical policies instead of guessing.

## 6. Why LangGraph / LangChain Used
LangChain and LangGraph are specialized frameworks designed to build agents. Instead of writing extremely messy manual code, we use LangGraph to define a structured workflow (called a State Graph). We create distinct "nodes"—one node for retrieving documents, and another for generating the report. This modularity means we can easily add more steps later (like sending an email) without breaking the system.

## 7. How the Agent Makes Decisions
The agent relies heavily on the "Prompt." By forcing the LLM to output specific sections (`Borrower Profile Summary`, `Credit Risk Analysis`, `Recommended Lending Decision`, etc.), the agent is cornered into creating a highly restricted, professional decision format. It cross-references the ML probability against the text it retrieved from the vector database.

## 8. How Bias is Reduced & Ethical Lending Ensured
The system handles bias in two ways:
1. **Explainability**: The agent must justify out loud *why* it recommends rejection.
2. **Ethical Framing**: Our knowledge base (the text rules) explicitly instructs the agent that algorithmic recommendations are not absolute and that elements like "age" shouldn't be single-factor discriminators. If confidence is low or fields are messy, the prompt forces the agent to recommend "Further Review Required" instead of a flat rejection.

## 9. How Structured Output Works
We use Prompt Engineering constraints to ensure the output remains orderly. We give the LLM template headers (using markdown like `##`) and instruct it strictly to format its response according to those headers. 

## 10. Why Streamlit is Chosen
Streamlit is used because it translates Python scripts directly into interactive web applications. It removes the need for beginner students to learn HTML, CSS, JavaScript, or complex backend routing architectures, keeping everything within standard Python logic.

## 11. Deployment Instructions (Streamlit Cloud)
To deploy this publicly for free:
1. Push all your code (including `requirements.txt`) to a public GitHub repository. (Note: **Do not** push your `GROQ_API_KEY`, which is why the Sidebar allows users to insert it dynamically!)
2. Go to [share.streamlit.io](https://share.streamlit.io) and create an account using your GitHub.
3. Click **"New App"**, and select your credit risk repository.
4. Set the main file path to `app.py`.
5. Click **Deploy**. Streamlit Cloud will automatically read `requirements.txt`, install Langchain, SentenceTransformers, and all dependencies, and host your project on the web within minutes!

## 12. Future Improvements
- Add more exhaustive regulation text files (e.g. 50+ page PDFs) into the Document Loader.
- Allow users to upload CSVs in Streamlit for bulk approvals.
- Use a local Ollama LLM setup instead of an API to make the data 100% private and offline.
