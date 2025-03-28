from typing import Dict, Any

class Singleton:
    """
    Singleton pattern ensures a class has only one instance and provides a global point
    of access to that instance.
    """
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize only if config is empty
        if not self._config:
            self._config = {
                "app_name": "LangGraph Framework",
                "version": "1.0.0",
                "debug": False
            }

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration dictionary."""
        return self._config

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the configuration with new values."""
        self._config.update(new_config)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        return self._config.get(key, default)

    def set_config_value(self, key: str, value: Any) -> None:
        """Set a specific configuration value."""
        self._config[key] = value

def main():
    # Test the Singleton pattern
    config1 = Singleton()
    config2 = Singleton()

    print("Testing Singleton pattern:")
    print(f"Are instances the same? {config1 is config2}")
    
    # Modify config through first instance
    config1.set_config_value("debug", True)
    
    # Check value through second instance
    print(f"Debug value from second instance: {config2.get_config_value('debug')}")
    
    # Update multiple values
    config1.update_config({
        "new_setting": "test",
        "version": "1.0.1"
    })
    
    # Print full config from second instance
    print("Full config from second instance:", config2.config)

if __name__ == "__main__":
    main() 