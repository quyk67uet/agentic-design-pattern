from abc import ABC, abstractmethod
from typing import Dict, Optional
import time

class ImageInterface(ABC):
    """Abstract interface for image operations."""
    
    @abstractmethod
    def display(self) -> None:
        """Display the image."""
        pass

    @abstractmethod
    def get_filename(self) -> str:
        """Get the image filename."""
        pass

class RealImage(ImageInterface):
    """Real image object that is expensive to create."""
    
    def __init__(self, filename: str):
        self._filename = filename
        self._load_image()

    def _load_image(self) -> None:
        """Simulate loading image from disk."""
        print(f"Loading image: {self._filename}")
        # Simulate expensive operation
        time.sleep(1)

    def display(self) -> None:
        """Display the image."""
        print(f"Displaying image: {self._filename}")

    def get_filename(self) -> str:
        """Get the image filename."""
        return self._filename

class ImageProxy(ImageInterface):
    """
    Proxy for RealImage that implements lazy loading and caching.
    Also provides access control and logging.
    """
    
    def __init__(self, filename: str):
        self._filename = filename
        self._real_image: Optional[RealImage] = None
        self._access_count: int = 0
        self._cache: Dict[str, str] = {}

    def _check_access(self) -> bool:
        """
        Check if access is allowed.
        Could implement more complex access control logic.
        """
        self._access_count += 1
        print(f"Access count for {self._filename}: {self._access_count}")
        return True

    def _log_access(self) -> None:
        """Log access to the image."""
        print(f"Logging access to image: {self._filename}")

    def display(self) -> None:
        """Display the image, creating it only if necessary."""
        if not self._check_access():
            print("Access denied")
            return

        # Lazy loading
        if self._real_image is None:
            self._real_image = RealImage(self._filename)

        # Check cache
        if self._filename in self._cache:
            print(f"Displaying cached image: {self._filename}")
            return

        self._real_image.display()
        self._cache[self._filename] = "cached"
        self._log_access()

    def get_filename(self) -> str:
        """Get the image filename."""
        return self._filename

def main():
    # Create image proxies
    image1 = ImageProxy("photo1.jpg")
    image2 = ImageProxy("photo2.jpg")

    # First access - will load the image
    print("\nFirst access to image1:")
    image1.display()

    # Second access - will use cached version
    print("\nSecond access to image1:")
    image1.display()

    # Access to different image
    print("\nFirst access to image2:")
    image2.display()

    # Check filenames
    print("\nFilenames:")
    print(f"Image1: {image1.get_filename()}")
    print(f"Image2: {image2.get_filename()}")

if __name__ == "__main__":
    main() 