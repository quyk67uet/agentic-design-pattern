from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Observer(ABC):
    """Abstract base class for observers."""
    
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> None:
        """Update method called when subject state changes."""
        pass

class Subject(ABC):
    """Abstract base class for subjects."""
    
    def __init__(self):
        self._observers: List[Observer] = []
        self._state: Dict[str, Any] = {}

    def attach(self, observer: Observer) -> None:
        """Attach an observer to the subject."""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        """Detach an observer from the subject."""
        self._observers.remove(observer)

    def notify(self) -> None:
        """Notify all observers about state change."""
        for observer in self._observers:
            observer.update(self._state)

class WeatherStation(Subject):
    """Concrete subject (Observable) representing a weather station."""

    def set_measurements(self, temperature: float, humidity: float, pressure: float) -> None:
        """Set new weather measurements."""
        self._state = {
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure
        }
        self.notify()

class TemperatureDisplay(Observer):
    """Concrete observer that displays temperature."""

    def update(self, data: Dict[str, Any]) -> None:
        """Update display with new temperature."""
        temperature = data.get("temperature")
        print(f"Temperature Display: {temperature}°C")

class HumidityDisplay(Observer):
    """Concrete observer that displays humidity."""

    def update(self, data: Dict[str, Any]) -> None:
        """Update display with new humidity."""
        humidity = data.get("humidity")
        print(f"Humidity Display: {humidity}%")

class WeatherForecast(Observer):
    """Concrete observer that makes weather predictions."""

    def update(self, data: Dict[str, Any]) -> None:
        """Update forecast based on new weather data."""
        temperature = data.get("temperature", 0)
        humidity = data.get("humidity", 0)
        pressure = data.get("pressure", 0)

        # Simple forecast logic
        if temperature > 25 and humidity > 70:
            print("Forecast: Expect thunderstorms")
        elif temperature > 30:
            print("Forecast: Hot and sunny")
        elif pressure < 1000:
            print("Forecast: Rain likely")
        else:
            print("Forecast: Fair weather")

def main():
    # Create the weather station (subject)
    weather_station = WeatherStation()

    # Create displays (observers)
    temp_display = TemperatureDisplay()
    humidity_display = HumidityDisplay()
    forecaster = WeatherForecast()

    # Attach observers to the weather station
    weather_station.attach(temp_display)
    weather_station.attach(humidity_display)
    weather_station.attach(forecaster)

    print("First weather update:")
    weather_station.set_measurements(24.5, 65, 1013)

    print("\nSecond weather update:")
    weather_station.set_measurements(32.5, 75, 995)

    # Detach an observer
    print("\nDetaching humidity display...")
    weather_station.detach(humidity_display)

    print("\nThird weather update:")
    weather_station.set_measurements(28.0, 80, 1005)

if __name__ == "__main__":
    main() 