from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class Handler(ABC):
    """Abstract base handler in the chain of responsibility."""
    
    def __init__(self):
        self._next_handler: Optional[Handler] = None

    def set_next(self, handler: 'Handler') -> 'Handler':
        """Set the next handler in the chain."""
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        """Handle the request or pass it to the next handler."""
        pass

class AuthenticationHandler(Handler):
    """Handles user authentication."""
    
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        """Check if user is authenticated."""
        if not request.get("authenticated"):
            return "Error: User is not authenticated"
        
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

class AuthorizationHandler(Handler):
    """Handles user authorization."""
    
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        """Check if user has required permissions."""
        user_role = request.get("role", "")
        required_role = request.get("required_role", "")
        
        if user_role != required_role:
            return f"Error: User role '{user_role}' does not have permission. Required role: '{required_role}'"
        
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

class ValidationHandler(Handler):
    """Handles input validation."""
    
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        """Validate request data."""
        data = request.get("data", {})
        
        if not isinstance(data, dict):
            return "Error: Invalid data format"
            
        required_fields = ["name", "email"]
        for field in required_fields:
            if field not in data:
                return f"Error: Missing required field '{field}'"
            
        if not self._is_valid_email(data["email"]):
            return "Error: Invalid email format"
        
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

    def _is_valid_email(self, email: str) -> bool:
        """Simple email validation."""
        return "@" in email and "." in email

class ProcessingHandler(Handler):
    """Handles the actual request processing."""
    
    def handle(self, request: Dict[str, Any]) -> Optional[str]:
        """Process the request if all previous checks passed."""
        data = request.get("data", {})
        return f"Success: Request processed for user {data['name']}"

def create_request_chain() -> Handler:
    """Create and configure the chain of handlers."""
    auth = AuthenticationHandler()
    authz = AuthorizationHandler()
    validation = ValidationHandler()
    processing = ProcessingHandler()

    auth.set_next(authz).set_next(validation).set_next(processing)
    return auth

def main():
    # Create the chain
    chain = create_request_chain()

    # Test case 1: Unauthenticated request
    print("Test 1: Unauthenticated request")
    request1 = {
        "authenticated": False,
        "role": "user",
        "required_role": "user",
        "data": {"name": "John", "email": "john@example.com"}
    }
    print(chain.handle(request1))

    # Test case 2: Unauthorized request
    print("\nTest 2: Unauthorized request")
    request2 = {
        "authenticated": True,
        "role": "user",
        "required_role": "admin",
        "data": {"name": "John", "email": "john@example.com"}
    }
    print(chain.handle(request2))

    # Test case 3: Invalid data
    print("\nTest 3: Invalid data")
    request3 = {
        "authenticated": True,
        "role": "admin",
        "required_role": "admin",
        "data": {"name": "John", "email": "invalid-email"}
    }
    print(chain.handle(request3))

    # Test case 4: Valid request
    print("\nTest 4: Valid request")
    request4 = {
        "authenticated": True,
        "role": "admin",
        "required_role": "admin",
        "data": {"name": "John", "email": "john@example.com"}
    }
    print(chain.handle(request4))

if __name__ == "__main__":
    main() 