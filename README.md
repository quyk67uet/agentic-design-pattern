# Design Patterns Framework

This project implements various design patterns in Python, organized into three main categories:

## Creational Patterns

### 1. Singleton Pattern (`creational/singleton.py`)
- Ensures a class has only one instance
- Provides a global point of access to that instance
- Used for managing configuration and shared resources

### 2. Factory Pattern (`creational/factory.py`)
- Creates objects without exposing instantiation logic
- Refers to the newly created object through a common interface
- Implements CNN and MLP model creation

## Structural Patterns

### 1. Adapter Pattern (`structural/adapter.py`)
- Allows incompatible interfaces to work together
- Wraps an object in an adapter to make it compatible with another class
- Implements data source adapters for JSON and XML

### 2. Proxy Pattern (`structural/proxy.py`)
- Provides a surrogate or placeholder for another object
- Controls access to the original object
- Implements lazy loading and caching for image loading

## Behavioral Patterns

### 1. Observer Pattern (`behavioral/observer.py`)
- Defines a one-to-many dependency between objects
- When one object changes state, all its dependents are notified
- Implements a weather station monitoring system

### 2. Chain of Responsibility Pattern (`behavioral/chain_of_responsibility.py`)
- Passes requests along a chain of handlers
- Each handler decides to process or pass to the next
- Implements request processing with authentication, authorization, and validation

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your OpenAI API key (if needed):
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Each pattern has its own example implementation and can be run independently:

```bash
# Run Singleton example
python creational/singleton.py

# Run Factory example
python creational/factory.py

# Run Adapter example
python structural/adapter.py

# Run Proxy example
python structural/proxy.py

# Run Observer example
python behavioral/observer.py

# Run Chain of Responsibility example
python behavioral/chain_of_responsibility.py
```

## Project Structure

```
design_pattern/
├── creational/
│   ├── singleton.py
│   └── factory.py
├── structural/
│   ├── adapter.py
│   └── proxy.py
├── behavioral/
│   ├── observer.py
│   └── chain_of_responsibility.py
├── __init__.py
├── config.py
├── requirements.txt
└── README.md
``` 