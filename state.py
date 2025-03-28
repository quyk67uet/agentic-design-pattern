from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE

class State(TypedDict):
    """State definition for the conversation flow."""
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    intent: str  # User intent: "info", "action", "chat"
    response: str  # Final response

class IntentClassifier:
    """Class for classifying user intents."""
    
    @staticmethod
    def classify(message: str) -> str:
        """Classify the intent based on message content."""
        message = message.lower()
        if any(word in message for word in ["thông tin", "biết"]):
            return "info"
        elif any(word in message for word in ["làm", "thực hiện"]):
            return "action"
        return "chat"

class IntentHandler:
    """Handler class for different intents."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
    
    def handle_info(self, message: str) -> str:
        """Handle information requests."""
        response = self.llm.invoke([
            HumanMessage(content=f"Cung cấp thông tin: {message}")
        ])
        return response.content
    
    def handle_action(self, message: str) -> str:
        """Handle action requests."""
        response = self.llm.invoke([
            HumanMessage(content=f"Thực hiện hành động: {message}")
        ])
        return response.content
    
    def handle_chat(self, message: str) -> str:
        """Handle general chat."""
        response = self.llm.invoke([
            HumanMessage(content=f"Trò chuyện: {message}")
        ])
        return response.content

def create_graph() -> StateGraph:
    """Create and configure the state machine graph."""
    intent_handler = IntentHandler()
    
    def classify_intent(state: State) -> State:
        """Classify user intent."""
        last_message = state["messages"][-1].content
        return {"intent": IntentClassifier.classify(last_message)}
    
    def handle_info(state: State) -> State:
        """Handle info intent."""
        response = intent_handler.handle_info(state["messages"][-1].content)
        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }
    
    def handle_action(state: State) -> State:
        """Handle action intent."""
        response = intent_handler.handle_action(state["messages"][-1].content)
        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }
    
    def handle_chat(state: State) -> State:
        """Handle chat intent."""
        response = intent_handler.handle_chat(state["messages"][-1].content)
        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }
    
    def route_intent(state: State) -> Literal["handle_info", "handle_action", "handle_chat"]:
        """Route to appropriate handler based on intent."""
        return f"handle_{state['intent']}"
    
    # Build graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("classify_intent", classify_intent)
    graph_builder.add_node("handle_info", handle_info)
    graph_builder.add_node("handle_action", handle_action)
    graph_builder.add_node("handle_chat", handle_chat)
    
    # Add edges
    graph_builder.add_edge(START, "classify_intent")
    graph_builder.add_conditional_edges("classify_intent", route_intent)
    graph_builder.add_edge("handle_info", END)
    graph_builder.add_edge("handle_action", END)
    graph_builder.add_edge("handle_chat", END)
    
    return graph_builder.compile()

def run_graph(user_input: str) -> None:
    """Run the conversation graph with user input."""
    graph = create_graph()
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="values"
    )
    
    for event in events:
        if "response" in event and event["response"]:
            print("Phản hồi:", event["response"])

if __name__ == "__main__":
    print("Nhập 'thoát' để dừng.")
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == "thoát":
            print("Tạm biệt!")
            break
        run_graph(user_input) 