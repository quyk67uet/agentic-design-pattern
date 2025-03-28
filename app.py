import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Annotated, Literal, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE,
    CNN_CONFIG, MLP_CONFIG, IMAGE_SIZE
)
from factory import ModelFactory
from behavioral.observer import Observer, Subject
from behavioral.chain_of_responsibility import Handler
from structural.adapter import DataSource
from structural.proxy import ImageInterface
from creational.singleton import Singleton

# Data handling components
class MNISTDataAdapter(DataSource):
    """Adapter for MNIST dataset."""
    
    def __init__(self, subset_size: int = 1000):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self._train_data = None
        self._test_data = None
        self.subset_size = subset_size

    def get_data(self) -> Dict[str, DataLoader]:
        """Get MNIST data loaders with reduced size."""
        if not self._train_data:
            full_train_data = datasets.MNIST(
                './data', train=True, download=True,
                transform=self.transform
            )
            full_test_data = datasets.MNIST(
                './data', train=False,
                transform=self.transform
            )
            
            # Use only a subset of data
            indices = torch.randperm(len(full_train_data))[:self.subset_size]
            self._train_data = torch.utils.data.Subset(full_train_data, indices)
            
            test_indices = torch.randperm(len(full_test_data))[:self.subset_size//5]
            self._test_data = torch.utils.data.Subset(full_test_data, test_indices)

        train_loader = DataLoader(self._train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(self._test_data, batch_size=100)
        
        return {
            "train": train_loader,
            "test": test_loader
        }

    def save_data(self, data: List[Dict[str, Any]]) -> bool:
        """Not implemented for MNIST dataset."""
        return False

class TrainingObserver(Observer):
    """Observer for monitoring training progress."""
    
    def update(self, data: Dict[str, Any]) -> None:
        """Print training progress."""
        epoch = data.get("epoch", 0)
        loss = data.get("loss", 0.0)
        accuracy = data.get("accuracy", 0.0)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

class ModelTrainer(Subject):
    """Subject that handles model training."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self._load_pretrained()
    
    def _load_pretrained(self):
        """Try to load pretrained weights if available."""
        try:
            self.model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
            print("Loaded pretrained model")
        except:
            print("No pretrained model found, will train from scratch")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> None:
        """Train for one epoch with early stopping."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Only train on a few batches for demonstration
        max_batches = 10
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        accuracy = 100. * correct / total
        avg_loss = total_loss / min(len(train_loader), max_batches)
        
        # Save model if accuracy is good enough
        if accuracy > 90:
            torch.save(self.model.state_dict(), 'mnist_model.pth')
            print("Saved model checkpoint")
        
        self._state = {
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy
        }
        self.notify()

class PredictionCache(ImageInterface):
    """Proxy for caching model predictions."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._cache: Dict[str, int] = {}
        self._access_count: int = 0

    def predict(self, image: torch.Tensor) -> int:
        """Get prediction for an image, using cache if available."""
        image_key = str(image.numpy().tobytes())
        
        self._access_count += 1
        if image_key in self._cache:
            print(f"Using cached prediction (access #{self._access_count})")
            return self._cache[image_key]

        # Make new prediction
        print(f"Making new prediction (access #{self._access_count})")
        with torch.no_grad():
            output = self.model(image.to(self.device))
            prediction = output.argmax(dim=1).item()
            self._cache[image_key] = prediction
            return prediction

    def display(self) -> None:
        """Not implemented."""
        pass

    def get_filename(self) -> str:
        """Not implemented."""
        return ""

class ValidationHandler(Handler):
    """Validates input data."""
    
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        """Validate the input data."""
        data = request.get("data")
        if not isinstance(data, torch.Tensor):
            return "Error: Input must be a PyTorch tensor"
        
        if data.dim() != 4:  # [batch_size, channels, height, width]
            return "Error: Input must be a 4D tensor"
            
        if data.size(1) != 1:  # Single channel for MNIST
            return "Error: Input must be single-channel"
            
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

class State(TypedDict):
    """State definition for the application flow."""
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    intent: str  # Intent: "classify", "train", "chat"
    response: str  # Final response
    model: nn.Module  # PyTorch model

class IntentClassifier:
    """Class for classifying user intents."""
    
    @staticmethod
    def classify(message: str) -> str:
        """Classify the intent based on message content."""
        message = message.lower().strip()
        
        # Check for exact matches first
        if message in ["1", "huấn luyện", "train"]:
            return "train"
        if message in ["2", "phân loại", "classify", "dự đoán", "predict"]:
            return "classify"
        if message in ["3", "đổi mô hình", "chuyển mô hình", "switch model", "change model"] or \
           "chuyển sang" in message or "dùng mô hình" in message:
            return "switch_model"
            
        # Then check for partial matches
        if any(word in message for word in ["huấn luyện", "train"]):
            return "train"
        if any(word in message for word in ["phân loại", "classify", "dự đoán", "predict"]):
            return "classify"
            
        return "chat"

class IntentHandler:
    """Handler class for different intents."""
    
    def __init__(self, model: nn.Module, config: Singleton, factory: ModelFactory):
        self.model = model
        self.device = torch.device("cpu")  # Force CPU usage
        self.model.to(self.device)
        self.config = config
        self.factory = factory
        
        # Initialize components with smaller dataset
        self.data_adapter = MNISTDataAdapter(subset_size=1000)  # Use only 1000 samples
        self.trainer = ModelTrainer(model, self.device)
        self.prediction_cache = PredictionCache(model, self.device)
        self.validation_handler = ValidationHandler()
        
        # Add training observer
        self.training_observer = TrainingObserver()
        self.trainer.attach(self.training_observer)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )

    def handle_classify(self, message: str) -> str:
        """Handle image classification requests."""
        # Get a random test image
        test_loader = self.data_adapter.get_data()["test"]
        data, target = next(iter(test_loader))
        image = data[0]  # Get first image
        
        # Validate input
        validation_result = self.validation_handler.handle({"data": image.unsqueeze(0)})
        if validation_result:
            return validation_result
            
        # Get prediction using cache
        prediction = self.prediction_cache.predict(image.unsqueeze(0))
        return f"Ảnh được phân loại là số: {prediction}"

    def handle_train(self, message: str) -> str:
        """Handle model training requests."""
        print("Starting training...")  # Debug print
        data_loaders = self.data_adapter.get_data()
        train_loader = data_loaders["train"]
        
        num_epochs = 1  # For demonstration
        for epoch in range(num_epochs):
            self.trainer.train_epoch(train_loader, epoch)
        
        return "Mô hình đã được huấn luyện xong!"
        
    def handle_switch_model(self, message: str) -> str:
        """Handle model type switching requests."""
        message = message.lower()
        
        # Determine which model to switch to
        if "cnn" in message:
            new_model_type = "cnn"
        elif "mlp" in message:
            new_model_type = "mlp"
        else:
            # Toggle between CNN and MLP
            current_model_type = self.config.get_config_value("model_type")
            new_model_type = "mlp" if current_model_type == "cnn" else "cnn"
        
        # If already using this model type, no need to switch
        if new_model_type == self.config.get_config_value("model_type"):
            return f"Đã đang sử dụng mô hình {new_model_type.upper()}"
        
        # Update config
        self.config.update_config({"model_type": new_model_type})
        
        # Create new model
        model_config = CNN_CONFIG if new_model_type == "cnn" else MLP_CONFIG
        self.model = self.factory.create_model(new_model_type, **model_config)
        self.model.to(self.device)
        
        # Reinitialize components with new model
        self.trainer = ModelTrainer(self.model, self.device)
        self.prediction_cache = PredictionCache(self.model, self.device)
        
        # Re-attach observer
        self.trainer.attach(self.training_observer)
        
        return f"Đã chuyển sang sử dụng mô hình {new_model_type.upper()}"

    def handle_chat(self, message: str) -> str:
        """Handle general chat."""
        response = self.llm.invoke([
            HumanMessage(content=f"Trò chuyện: {message}")
        ])
        return response.content

def create_graph(intent_handler: IntentHandler) -> StateGraph:
    """Create and configure the state machine graph."""
    
    def classify_intent(state: State) -> State:
        """Classify user intent."""
        last_message = state["messages"][-1].content
        return {"intent": IntentClassifier.classify(last_message)}
    
    def handle_classify(state: State) -> State:
        """Handle classify intent."""
        response = intent_handler.handle_classify(state["messages"][-1].content)
        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }
    
    def handle_train(state: State) -> State:
        """Handle train intent."""
        response = intent_handler.handle_train(state["messages"][-1].content)
        return {
            "response": response,
            "messages": [AIMessage(content=response)]
        }
        
    def handle_switch_model(state: State) -> State:
        """Handle model switching intent."""
        response = intent_handler.handle_switch_model(state["messages"][-1].content)
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
    
    def route_intent(state: State) -> Literal["handle_classify", "handle_train", "handle_switch_model", "handle_chat"]:
        """Route to appropriate handler based on intent."""
        return f"handle_{state['intent']}"
    
    # Build graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("classify_intent", classify_intent)
    graph_builder.add_node("handle_classify", handle_classify)
    graph_builder.add_node("handle_train", handle_train)
    graph_builder.add_node("handle_switch_model", handle_switch_model)
    graph_builder.add_node("handle_chat", handle_chat)
    
    # Add edges
    graph_builder.add_edge(START, "classify_intent")
    graph_builder.add_conditional_edges("classify_intent", route_intent)
    graph_builder.add_edge("handle_classify", END)
    graph_builder.add_edge("handle_train", END)
    graph_builder.add_edge("handle_switch_model", END)
    graph_builder.add_edge("handle_chat", END)
    
    return graph_builder.compile()

def run_graph(user_input: str, intent_handler: IntentHandler) -> None:
    """Run the application graph with user input."""
    graph = create_graph(intent_handler)
    events = graph.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "model": intent_handler.model
        },
        {"configurable": {"thread_id": "1"}},
        stream_mode="values"
    )
    
    for event in events:
        if "response" in event and event["response"]:
            print("Phản hồi:", event["response"])

def main():
    """Main application entry point."""
    # Get configuration from singleton
    config = Singleton()
    config.update_config({
        "model_type": "cnn",  # Có thể là "cnn" hoặc "mlp"
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })
    
    # Get factory instance
    factory = ModelFactory()
    
    # Chọn config dựa trên model_type
    model_type = config.get_config_value("model_type")
    model_config = CNN_CONFIG if model_type == "cnn" else MLP_CONFIG
    
    # Create model using factory
    model = factory.create_model(
        model_type,
        **model_config
    )
    print(f"Đã tạo mô hình {model_type.upper()} "
          f"trên {config.get_config_value('device')}")
    
    # Create a single intent handler instance
    intent_handler = IntentHandler(model, config, factory)
    
    print("\nCác lệnh có sẵn:")
    print("1. 'huấn luyện' hoặc 'train': Huấn luyện mô hình")
    print("2. 'phân loại' hoặc 'classify': Phân loại một ảnh ngẫu nhiên")
    print("3. 'đổi mô hình' hoặc 'chuyển mô hình': Chuyển đổi giữa CNN và MLP")
    print("4. 'thoát': Kết thúc chương trình")
    print("5. Các câu khác: Trò chuyện với chatbot")
    
    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() == "thoát":
            print("Tạm biệt!")
            break
        run_graph(user_input, intent_handler)

if __name__ == "__main__":
    main() 