from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DataSource(ABC):
    """Abstract interface for data sources."""
    
    @abstractmethod
    def get_data(self) -> List[Dict[str, Any]]:
        """Get data from the source."""
        pass

    @abstractmethod
    def save_data(self, data: List[Dict[str, Any]]) -> bool:
        """Save data to the source."""
        pass

class JSONDataSource:
    """Legacy JSON data source with incompatible interface."""
    
    def __init__(self):
        self._data = []

    def read_json(self) -> List[Dict[str, Any]]:
        """Read data in JSON format."""
        return self._data

    def write_json(self, json_data: List[Dict[str, Any]]) -> None:
        """Write data in JSON format."""
        self._data = json_data

class XMLDataSource:
    """Legacy XML data source with incompatible interface."""
    
    def __init__(self):
        self._data = []

    def parse_xml(self) -> List[Dict[str, Any]]:
        """Parse XML data."""
        return self._data

    def generate_xml(self, data: List[Dict[str, Any]]) -> None:
        """Generate XML from data."""
        self._data = data

class JSONAdapter(DataSource):
    """Adapter for JSON data source."""
    
    def __init__(self, json_source: JSONDataSource):
        self._source = json_source

    def get_data(self) -> List[Dict[str, Any]]:
        """Adapt read_json to get_data."""
        return self._source.read_json()

    def save_data(self, data: List[Dict[str, Any]]) -> bool:
        """Adapt write_json to save_data."""
        try:
            self._source.write_json(data)
            return True
        except Exception:
            return False

class XMLAdapter(DataSource):
    """Adapter for XML data source."""
    
    def __init__(self, xml_source: XMLDataSource):
        self._source = xml_source

    def get_data(self) -> List[Dict[str, Any]]:
        """Adapt parse_xml to get_data."""
        return self._source.parse_xml()

    def save_data(self, data: List[Dict[str, Any]]) -> bool:
        """Adapt generate_xml to save_data."""
        try:
            self._source.generate_xml(data)
            return True
        except Exception:
            return False

def main():
    # Test data
    test_data = [
        {"id": 1, "name": "Item 1"},
        {"id": 2, "name": "Item 2"}
    ]

    # Create sources
    json_source = JSONDataSource()
    xml_source = XMLDataSource()

    # Create adapters
    json_adapter = JSONAdapter(json_source)
    xml_adapter = XMLAdapter(xml_source)

    # Test JSON adapter
    print("Testing JSON adapter:")
    json_adapter.save_data(test_data)
    retrieved_json = json_adapter.get_data()
    print(f"JSON data: {retrieved_json}")

    # Test XML adapter
    print("\nTesting XML adapter:")
    xml_adapter.save_data(test_data)
    retrieved_xml = xml_adapter.get_data()
    print(f"XML data: {retrieved_xml}")

if __name__ == "__main__":
    main() 