import torch
import torch.nn as nn
from typing import Dict, Any, Type
from config import CNN_CONFIG, MLP_CONFIG

class Model(nn.Module):
    """Base model class that all specific models should inherit from."""
    
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")

class CNN(Model):
    """Convolutional Neural Network implementation."""
    
    def __init__(self, **kwargs):
        super().__init__()
        config = {**CNN_CONFIG, **kwargs}
        
        self.conv1 = nn.Conv2d(
            config["input_channels"],
            config["hidden_size"],
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(config["hidden_size"] * 14 * 14, config["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MLP(Model):
    """Multi-Layer Perceptron implementation."""
    
    def __init__(self, **kwargs):
        super().__init__()
        config = {**MLP_CONFIG, **kwargs}
        
        self.fc1 = nn.Linear(config["input_size"], config["hidden_size"])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config["hidden_size"], config["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class ModelFactory:
    """Factory class for creating different types of neural network models."""
    
    _models: Dict[str, Type[Model]] = {
        "cnn": CNN,
        "mlp": MLP
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> Model:
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: Type of model to create ("cnn" or "mlp")
            **kwargs: Additional configuration parameters
            
        Returns:
            An instance of the specified model type
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return cls._models[model_type](**kwargs)

def main():
    """Test the factory pattern implementation."""
    # Create CNN model
    cnn_model = ModelFactory.create_model("cnn", **CNN_CONFIG)
    print("Đã tạo mô hình CNN:", cnn_model)

    # Create MLP model
    mlp_model = ModelFactory.create_model("mlp", **MLP_CONFIG)
    print("Đã tạo mô hình MLP:", mlp_model)

    # Test with dummy input
    dummy_input = torch.randn(1, 1, 28, 28)  # Batch_size=1, Channels=1, Height=28, Width=28
    cnn_output = cnn_model(dummy_input)
    print("Đầu ra CNN:", cnn_output.shape)

    mlp_output = mlp_model(dummy_input)
    print("Đầu ra MLP:", mlp_output.shape)

if __name__ == "__main__":
    main() 