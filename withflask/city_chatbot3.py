import os
import json
import datetime # No longer needed
import re # Keep for potential input cleaning
from typing import TypedDict, Annotated, Dict, Optional, Literal, List, Union

# --- RAG and Core LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# For loading environment variables (recommended for API keys)
from dotenv import load_dotenv

# --- Constants ---
FEEDBACK_FILE = "city_feedback.json" # Keep for feedback node
VOTES_FILE = "city_votes.json"       # Keep for vote node
GRAPH_IMAGE_FILE = "arlington_app_qa_feedback_graph.png" # Output file name

# --- RAG Constants ---
EMBEDDING_MODEL = "models/embedding-001"
PERSIST_DIRECTORY = "./chroma-db"
VECTOR_STORE_NAME = "simple-rag"

# --- API Key and LLM/Retriever Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = None
retriever = None

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not found.")
    exit()
else:
     try:
         print("Initializing Gemini LLM...")
         llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
         print("Gemini LLM Initialized.")

         print("Initializing Embeddings and Retriever...")
         embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
         vector_db = Chroma(
             embedding_function=embedding,
             collection_name=VECTOR_STORE_NAME,
             persist_directory=PERSIST_DIRECTORY,
         )
         retriever = vector_db.as_retriever(search_kwargs={"k": 3})
         print("Retriever Initialized.")

     except Exception as e:
        print(f"Error during initialization: {e}")
        exit()

# --- State Definition (Q&A + Feedback/Vote) ---
class CityAppState(TypedDict):
    """Represents the state of the Arlington Q&A + Feedback workflow."""
    citizen_id: str

    # Q&A Part
    user_question: Optional[str]
    Youtube: Optional[str]

    # Feedback/Vote Part
    feedback: Dict[str, str]
    votes: Dict[str, str]

    # Conversation history & Control
    messages: Annotated[list, add_messages]
    next_expected_input: Optional[Literal[
        'initial_question', 'feedback_consent', 'feedback_text',
        'vote_consent', 'vote_choice'
    ]]

# --- JSON Helper Functions (load_data, save_data - keep) ---
def load_data(filepath: str) -> dict:
    if not os.path.exists(filepath): return {}
    try:
        with open(filepath, 'r') as f: content = f.read()
        return json.loads(content) if content else {}
    except Exception as e: print(f"Error loading {filepath}: {e}"); return {}

def save_data(filepath: str, data: dict):
    try:
        with open(filepath, 'w') as f: json.dump(data, f, indent=4)
        print(f"Data successfully saved to {filepath}")
    except Exception as e: print(f"Error saving {filepath}: {e}")


# --- Node Functions ---

def prompt_initial_question(state: CityAppState) -> dict:
    """Asks the user for their question about Arlington."""
    print("--- Node: prompt_initial_question ---")
    ai_message = AIMessage(content="Welcome! Please ask your question about Arlington (e.g., 'What is the city budget for parks?', 'Tell me about recent zoning changes.', or type 'quit').")
    return {"messages": [ai_message], "next_expected_input": "initial_question"}

def answer_question_rag(state: CityAppState) -> dict:
    """Answers the user's question using RAG."""
    print("--- Node: answer_question_rag ---")
    user_query = state.get("user_question")
    if not user_query:
        return {"Youtube": "Error: Missing question.", "messages": [AIMessage(content="I seem to have missed your question.")]}

    if not retriever or not llm:
        return {"Youtube": "Error: Cannot process request.", "messages": [AIMessage(content="Sorry, internal error.")]}

    print(f"Answering query: '{user_query}'")
    template = """You are a helpful assistant answering questions about Arlington, TX based *only* on the provided context. If the context doesn't contain the answer, state that clearly. Be concise.
Context:
{context}
Question:
{question}
Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    try:
        answer = rag_chain.invoke(user_query)
        print(f"Generated Answer: {answer[:200]}...")
        return {"Youtube": answer}
    except Exception as e:
        print(f"Error during RAG processing: {e}")
        return {"Youtube": "Sorry, an error occurred answering.", "messages": [AIMessage(content="Error processing question.")]}

def display_answer(state: CityAppState) -> dict:
     """Displays the answer to the question."""
     print("--- Node: display_answer ---")
     answer = state.get("Youtube", "Sorry, I couldn't generate an answer.")
     ai_message = AIMessage(content=answer)
     # Don't set next_expected_input here, graph edge goes to ask_feedback
     return {"messages": [ai_message]}

# --- Feedback Nodes ---
def ask_feedback(state: CityAppState) -> dict:
    print("--- Node: ask_feedback ---")
    # Ask for feedback on the general answer provided
    if not state.get("Youtube") or "error" in state.get("Youtube","").lower():
        print("Skipping feedback as answer wasn't generated successfully.")
        return {"next_expected_input": None} # Skip to vote directly

    ai_message = AIMessage(content="Was this answer helpful? Would you like to provide feedback? (yes/no)")
    return {"messages": [ai_message], "next_expected_input": "feedback_consent"}

def prompt_for_feedback_text(state: CityAppState) -> dict:
    print("--- Node: prompt_for_feedback_text ---")
    ai_message = AIMessage(content="Okay, please type your feedback below:")
    return {"messages": [ai_message], "next_expected_input": "feedback_text"}

def store_feedback(state: CityAppState) -> dict:
    print("--- Node: store_feedback ---")
    citizen_id = state.get("citizen_id", "UNKNOWN")
    if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage): return {}
    user_feedback = state["messages"][-1].content
    if user_feedback.lower() == "quit": return {}
    if citizen_id == "UNKNOWN": return {}

    # Store feedback perhaps linked to the question?
    question = state.get("user_question", "Unknown question")
    feedback_data = {
        "question": question,
        "answer": state.get("Youtube", "N/A"),
        "feedback_text": user_feedback,
        "timestamp": datetime.datetime.now().isoformat() # Add timestamp
    }

    feedback_dict = load_data(FEEDBACK_FILE)
    if citizen_id not in feedback_dict:
        feedback_dict[citizen_id] = []
    feedback_dict[citizen_id].append(feedback_data) # Append feedback entry

    print(f"Feedback for Citizen ID {citizen_id} on question '{question[:50]}...' prepared.")
    save_data(FEEDBACK_FILE, feedback_dict)
    # Update state with the *entire* feedback structure for this user if needed later
    # Or just confirm success, no need to return full dict unless generate_final needs it
    # Let's return the specific entry added? No, just confirm.
    # We need to update the state's 'feedback' field if generate_final_response uses it directly
    return {"feedback": feedback_dict}


def confirm_feedback_receipt(state: CityAppState) -> dict:
    print("--- Node: confirm_feedback_receipt ---")
    ai_message = AIMessage(content="Thank you, your feedback has been recorded.")
    # Don't set next_expected_input here, graph edge goes to ask_vote
    return {"messages": [ai_message]}

# --- Vote Nodes (Asking a Generic Symbolic Question) ---
def ask_vote(state: CityAppState) -> dict:
    print("--- Node: ask_vote ---")
    # Ask a generic symbolic question, as there's no specific meeting context
    # Check if the initial Q&A was successful before asking
    if not state.get("Youtube") or "error" in state.get("Youtube","").lower():
        print("Skipping vote as initial Q&A failed.")
        return {"next_expected_input": None} # Skip to final response

    # Generic symbolic question:
    vote_topic = "your overall satisfaction with this information service session"
    ai_message = AIMessage(content=f"Finally, would you like to cast a symbolic vote on {vote_topic}? (yes/no)")
    return {"messages": [ai_message], "next_expected_input": "vote_consent"}

def prompt_for_vote_choice(state: CityAppState) -> dict:
    print("--- Node: prompt_for_vote_choice ---")
    # Adapt prompt for the generic topic
    ai_message = AIMessage(content="Okay, please rate your satisfaction (e.g., 'Satisfied', 'Neutral', 'Unsatisfied'):")
    return {"messages": [ai_message], "next_expected_input": "vote_choice"}

def store_vote(state: CityAppState) -> dict:
    print("--- Node: store_vote ---")
    citizen_id = state.get("citizen_id", "UNKNOWN")
    if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage): return {}
    user_vote_text = state["messages"][-1].content
    if user_vote_text.lower() == "quit": return {}
    if citizen_id == "UNKNOWN": return {}

    # Store vote linked to the session/generic topic
    question = state.get("user_question", "General Session")
    vote_data = {
        "topic": "Session Satisfaction", # Or link to question
        "vote": user_vote_text.strip(),
        "timestamp": datetime.datetime.now().isoformat()
    }

    votes_dict = load_data(VOTES_FILE)
    if citizen_id not in votes_dict:
        votes_dict[citizen_id] = []
    votes_dict[citizen_id].append(vote_data) # Append vote entry

    print(f"Vote for Citizen ID {citizen_id} on '{vote_data['topic']}' prepared.")
    save_data(VOTES_FILE, votes_dict)
    # Update state's 'votes' field
    return {"votes": votes_dict}


def confirm_vote_receipt(state: CityAppState) -> dict:
    print("--- Node: confirm_vote_receipt ---")
    ai_message = AIMessage(content="Thank you, your symbolic vote has been recorded.")
    # Don't set next_expected_input, graph edge goes to final response
    return {"messages": [ai_message]}


# --- get_user_input node remains the same ---
def get_user_input(state: CityAppState) -> dict:
    """Gets input from the user via the console."""
    print("--- Node: get_user_input ---")
    if state["messages"] and isinstance(state["messages"][-1], AIMessage):
         print(f"\nArlington: {state['messages'][-1].content}")
    elif state["messages"]:
         print(f"\n(Last message was: {type(state['messages'][-1]).__name__})")

    prompt_text = "You: "
    user_input = input(prompt_text).strip()

    # Store input as the general question field for router processing
    update_dict = {"user_question": user_input}

    if user_input.lower() in ['q', 'quit', 'exit']:
         print("Exiting application.")
         return {"messages": [HumanMessage(content="quit")], **update_dict} # Quit flag

    return {"messages": [HumanMessage(content=user_input)], **update_dict} # Add user message


# --- Final Response Node ---
def generate_final_response(state: CityAppState) -> dict:
    """Generates a final response summarizing the interaction."""
    print("--- Node: generate_final_response ---")
    citizen_id = state.get("citizen_id", "UNKNOWN")
    question = state.get("user_question") # Last question asked? Or initial? Ambiguous now.
    answer = state.get("Youtube")

    # Use the state's feedback/votes dict which should be updated by store nodes
    feedback_list = state.get("feedback", {}).get(citizen_id, [])
    feedback_recorded = bool(feedback_list) # Check if any feedback was stored for user
    votes_list = state.get("votes", {}).get(citizen_id, [])
    vote_recorded = bool(votes_list) # Check if any vote was stored for user

    parts = ["**Session Summary**", f"Citizen ID: {citizen_id}"]
    if question and answer and "error" not in answer.lower() :
         parts.append(f"\nLast Question: {question}")
         # parts.append(f"Answer Provided: {answer}") # Maybe too verbose
    elif question:
         parts.append(f"\nLast Question: {question} (Answer may have failed)")
    else:
         parts.append("\nNo primary question processed in this session.")

    parts.append(f"Feedback Recorded This Session: {'Yes' if feedback_recorded else 'No'}")
    parts.append(f"Vote Recorded This Session: {'Yes' if vote_recorded else 'No'}")

    parts.append("\nThank you for using the Arlington Information Service!")
    final_response = "\n".join(parts)
    ai_message = AIMessage(content=final_response)
    return {"messages": [ai_message]}


# --- Routing Logic ---
def route_after_input(state: CityAppState) -> str:
    """Routes after user input based on expected step."""
    print("--- Router: route_after_input ---")
    expected = state.get("next_expected_input")
    last_message_content = ""
    # Ensure messages list exists and is not empty before accessing
    if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        last_message_content = state["messages"][-1].content.lower().strip()

    if last_message_content == "quit":
        print("Routing to END (quit command)")
        return END

    print(f"Routing based on expected input: {expected}")

    if expected == "initial_question":
        return "answer_question_rag"
    elif expected == "feedback_consent":
        return "prompt_for_feedback_text" if last_message_content.startswith('y') else "ask_vote"
    elif expected == "feedback_text":
        return "store_feedback"
    elif expected == "vote_consent":
        return "prompt_for_vote_choice" if last_message_content.startswith('y') else "generate_final_response"
    elif expected == "vote_choice":
        return "store_vote"
    else:
        # If expected is None (e.g. after confirmations), or unexpected
        print(f"Unexpected or None expected state '{expected}'. Routing to final response.")
        # This path might be hit after confirmations if graph isn't explicit
        # Defaulting to final response is safer than erroring or looping unexpectedly
        return "generate_final_response"


# --- Build the Graph ---
def create_city_graph():
    """Creates the LangGraph StateGraph for Q&A + Feedback/Vote."""
    graph_builder = StateGraph(CityAppState)

    # Add nodes
    graph_builder.add_node("prompt_initial_question", prompt_initial_question)
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("answer_question_rag", answer_question_rag)
    graph_builder.add_node("display_answer", display_answer)
    graph_builder.add_node("ask_feedback", ask_feedback)
    graph_builder.add_node("prompt_for_feedback_text", prompt_for_feedback_text)
    graph_builder.add_node("store_feedback", store_feedback)
    graph_builder.add_node("confirm_feedback_receipt", confirm_feedback_receipt)
    graph_builder.add_node("ask_vote", ask_vote)
    graph_builder.add_node("prompt_for_vote_choice", prompt_for_vote_choice)
    graph_builder.add_node("store_vote", store_vote)
    graph_builder.add_node("confirm_vote_receipt", confirm_vote_receipt)
    graph_builder.add_node("generate_final_response", generate_final_response)

    # Define Edges
    graph_builder.set_entry_point("prompt_initial_question")
    graph_builder.add_edge("prompt_initial_question", "get_user_input")

    # After getting input, route
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "answer_question_rag": "answer_question_rag",
            "prompt_for_feedback_text": "prompt_for_feedback_text",
            "ask_vote": "ask_vote", # Route directly if skipping feedback
            "store_feedback": "store_feedback",
            "prompt_for_vote_choice": "prompt_for_vote_choice",
            "generate_final_response": "generate_final_response", # Route directly if skipping vote
            "store_vote": "store_vote",
             END: END # Handle quit command from router
        }
    )

    # After answering Q -> Display -> Ask Feedback
    # Conditional check: If answer failed, skip feedback/vote
    def route_after_answer(state: CityAppState):
         if state.get("Youtube") and "error" not in state.get("Youtube","").lower():
              return "answer_ok"
         else:
              return "answer_failed"

    graph_builder.add_conditional_edges(
        "answer_question_rag",
        route_after_answer,
        {
            "answer_ok": "display_answer",
            "answer_failed": "generate_final_response" # Go to end if RAG failed
        }
    )
    graph_builder.add_edge("display_answer", "ask_feedback") # Always ask feedback after displaying good answer

    # Feedback Path
    graph_builder.add_edge("ask_feedback", "get_user_input")
    graph_builder.add_edge("prompt_for_feedback_text", "get_user_input")
    graph_builder.add_edge("store_feedback", "confirm_feedback_receipt")
    graph_builder.add_edge("confirm_feedback_receipt", "ask_vote") # After feedback confirm, ask vote

    # Vote Path
    graph_builder.add_edge("ask_vote", "get_user_input")
    graph_builder.add_edge("prompt_for_vote_choice", "get_user_input")
    graph_builder.add_edge("store_vote", "confirm_vote_receipt")
    graph_builder.add_edge("confirm_vote_receipt", "generate_final_response") # After vote confirm, finish

    # Final edge
    graph_builder.add_edge("generate_final_response", END)

    # Compile
    print("Compiling graph...")
    compiled_graph = graph_builder.compile()
    print("Graph compiled.")
    return compiled_graph

# --- Main Execution Logic ---
if __name__ == "__main__":
    if not llm or not retriever:
        print("LLM or Retriever not initialized. Exiting.")
        exit()

    print("\n--- Starting Arlington Q&A + Feedback Application (LangGraph) ---")

    # --- Get Citizen ID ---
    valid_id_entered = False; user_citizen_id = None
    while not valid_id_entered:
        user_citizen_id = input("Welcome! Please enter your two-digit Citizen ID: ").strip()
        if len(user_citizen_id) == 2 and user_citizen_id.isdigit(): valid_id_entered = True
        else: print("Invalid ID format.")
    print(f"Citizen ID {user_citizen_id} accepted.")

    # Initial state for the Q&A -> Feedback -> Vote flow
    initial_state = CityAppState(
        citizen_id=user_citizen_id,
        user_question=None,
        Youtube=None,
        feedback={}, # Initialize as empty dict
        votes={},    # Initialize as empty dict
        messages=[],
        next_expected_input='initial_question' # Start flow
    )

    print(f"\n--- Starting LangGraph workflow for Citizen {initial_state['citizen_id']} ---")
    arlington_app_graph = create_city_graph()

    # --- Optional: Save graph visualization ---
    try:
       png_data = arlington_app_graph.get_graph().draw_mermaid_png()
       with open(GRAPH_IMAGE_FILE, "wb") as f: f.write(png_data)
       print(f"Graph diagram saved to {GRAPH_IMAGE_FILE}")
    except Exception as e: print(f"Could not save graph diagram: {e}.")

    # --- Run the graph ---
    print("\n--- Running Graph ---")
    current_state = initial_state.copy()
    try:
        for event in arlington_app_graph.stream(current_state, {"recursion_limit": 30}):
            last_node = list(event.keys())[0]
            current_state = event[last_node]
            print(f"--- Completed Node: {last_node} ---")

        print("\n--- Run Complete ---")

    except Exception as e:
        print(f"\n--- An error occurred during execution: {e} ---")
        import traceback; traceback.print_exc()