import os
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from src.rag import get_retriever

# Define State for the LangGraph
class GraphState(TypedDict):
    borrower_data: dict
    ml_prediction_label: str
    ml_probability: float
    retrieved_context: str
    final_report: str

def retrieve_guidelines(state: GraphState):
    """
    Node function: Retrieves relevant ethical and lending guidelines based on the borrower's profile.
    """
    print("Agent Step: Retrieving guidelines...")
    retriever = get_retriever()
    
    # Formulate a quick query based on the borrower data
    debt_ratio = state["borrower_data"].get("debt_ratio", 0)
    monthly_inc = state["borrower_data"].get("monthly_inc", 0)
    risk_label = state["ml_prediction_label"]

    query = f"Guidelines for borrower with debt ratio {debt_ratio}, income {monthly_inc}, and model prediction {risk_label}."
    
    docs = retriever.invoke(query)
    context_text = "\\n\\n".join([doc.page_content for doc in docs])
    
    return {"retrieved_context": context_text}

def generate_report(state: GraphState):
    """
    Node function: Generates the structured lending decision report using a LLM.
    """
    print("Agent Step: Generating report...")
    
    # Initialize the LLM (Requires GROQ_API_KEY to be set in environment)
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    except Exception as e:
        return {"final_report": f"Error initializing LLM: {str(e)}\\nPlease check if your GROQ_API_KEY is properly set."}

    prompt_text = (
        "You are an expert AI Lending Assistant.\\n"
        "Your task is to generate a structured credit lending report for a borrower based on their profile, \\n"
        "the prediction of our Machine Learning model, and the retrieved ethical lending guidelines.\\n\\n"
        "Constraints:\\n"
        "1. Provide a structured report exactly matching the sections described below.\\n"
        "2. Form a recommendation of Approve, Reject, or Further Review Required.\\n"
        "3. If probability of default is around the threshold or missing data is prominent, recommend Further Review Required.\\n"
        "4. Reference the retrieved lending guidelines explicitly in your statements.\\n"
        "5. Provide a legal and ethical disclaimer at the end.\\n"
        "6. Make sure it's formatting using Markdown.\\n\\n"
        "--- Borrower Profile ---\\n"
        "{borrower_data}\\n\\n"
        "--- Machine Learning Model Context ---\\n"
        "Prediction: {ml_prediction_label}\\n"
        "Probability of Default: {ml_probability}\\n\\n"
        "--- Retrieved Guidelines and Policies ---\\n"
        "{retrieved_context}\\n\\n"
        "--- Required Structured Output Sections ---\\n"
        "## Borrower Profile Summary\\n"
        "## Credit Risk Analysis\\n"
        "## Recommended Lending Decision\\n"
        "## Risk Mitigation Suggestions\\n"
        "## Supporting References\\n"
        "## Legal and Ethical Disclaimer\\n"
    )

    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["borrower_data", "ml_prediction_label", "ml_probability", "retrieved_context"]
    )
    
    chain = prompt | llm
    
    response = chain.invoke({
        "borrower_data": str(state["borrower_data"]),
        "ml_prediction_label": state["ml_prediction_label"],
        "ml_probability": f"{state['ml_probability']:.2%}",
        "retrieved_context": state["retrieved_context"]
    })
    
    return {"final_report": response.content}

# Create the LangGraph
from langgraph.graph import StateGraph, END

def build_agent_graph():
    """
    Constructs the LangGraph connecting retrieval and generation nodes.
    """
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_guidelines)
    workflow.add_node("generate", generate_report)
    
    # Set entry point and edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile graph
    app = workflow.compile()
    return app

def run_agent(borrower_data, ml_prediction_label, ml_probability):
    """
    Main entry point to execute the agent.
    """
    # Setup state
    initial_state = {
        "borrower_data": borrower_data,
        "ml_prediction_label": ml_prediction_label,
        "ml_probability": ml_probability,
        "retrieved_context": "",
        "final_report": ""
    }
    
    app = build_agent_graph()
    
    # Run graph
    result = app.invoke(initial_state)
    return result["final_report"]
