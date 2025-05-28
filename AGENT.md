# AGENT.md - Flock Framework Onboarding Guide

## Project Overview

**Flock** is a declarative AI agent orchestration framework built by white duck GmbH. It solves common LLM development pain points by providing:

- **Declarative Contracts**: Define inputs/outputs with Pydantic models instead of brittle prompts
- **Built-in Resilience**: Automatic retries, state persistence via Temporal.io
- **Production-Ready**: Deploy as REST APIs, scale without rewriting
- **Actually Testable**: Clear contracts make agents unit-testable
- **Dynamic Workflows**: Self-correcting loops, conditional routing

**Key Differentiator**: You define what goes in and what should come out - the framework handles the "how" with LLMs.

## Project Structure

```
flock/
├── src/flock/
│   ├── core/                   # Framework foundation
│   │   ├── flock.py           # Main orchestrator class
│   │   ├── flock_agent.py     # Base agent class (1000+ lines)
│   │   ├── flock_registry.py  # Component discovery & registration
│   │   ├── context/           # State management
│   │   ├── execution/         # Local & Temporal executors
│   │   ├── serialization/     # Save/load functionality
│   │   └── mcp/              # Model Context Protocol integration
│   ├── evaluators/            # Agent execution logic (DSPy-based)
│   ├── modules/               # Pluggable behavior extensions
│   ├── tools/                 # Utility functions
│   ├── webapp/                # FastAPI web interface
│   └── workflow/              # Temporal.io activities
├── tests/                     # Comprehensive test suite
├── examples/                  # Usage examples and showcases
└── docs/                      # Documentation
```

## Key Components & Architecture

### Core Classes

1. **`Flock`** (`src/flock/core/flock.py`)
   - Main orchestrator, manages agents and execution
   - Handles both local and Temporal.io execution
   - Entry point for most operations

2. **`FlockAgent`** (`src/flock/core/flock_agent.py`)
   - Base class for all agents (refactored from 1000+ to 263 lines)
   - Lifecycle hooks: initialize → evaluate → terminate
   - Supports modules, evaluators, routers
   - Uses composition-based architecture with focused components

3. **`FlockRegistry`** (`src/flock/core/flock_registry.py`)
   - Singleton for component discovery
   - Manages agents, callables, types, servers
   - Auto-registration capabilities

4. **`FlockContext`** (`src/flock/core/context/context.py`)
   - State management between agent executions
   - History tracking, variable storage

### Execution Flow

```
Flock.run() → FlockAgent.run_async() → Evaluator.evaluate() → Router.route() → Next Agent
```

## Development Workflow

### Essential Commands

```bash
# Project uses UV as package manager
uv run python -m pytest tests/core/test_flock_core.py -v    # Run core tests
uv run python -m pytest tests/serialization/ -v            # Test serialization
uv run python -m pytest tests/modules/ -v -k memory         # Test specific modules

# Common development tasks
uv run python examples/01-getting-started/quickstart.py     # Run basic example
uv run python -c "from flock.core import Flock; print('OK')" # Quick import test
```

### Testing Strategy

- **Unit Tests**: `tests/core/` for framework components
- **Integration Tests**: `tests/integration/` for external dependencies  
- **Module Tests**: `tests/modules/` for specific functionality
- **Serialization Tests**: `tests/serialization/` for save/load

**Important**: Many tests currently have issues unrelated to core functionality (logging conflicts, registry state). Focus on functionality tests.

## Known Issues & Gotchas

### Current Problems
1. **Logging conflicts**: `exc_info` parameter duplication causing test failures
2. **Registry state**: Global singleton causes test isolation issues
3. **Test brittleness**: Some tests depend on external services or configuration

### Code Quality Issues Found
- Bare `except:` handlers in multiple files
- Global state management patterns
- Some circular import dependencies
- Complex function complexity (Ruff warnings)

## Important Patterns & Conventions

### Component Registration
```python
from flock.core.flock_registry import flock_component

@flock_component(config_class=MyComponentConfig)
class MyComponent(FlockEvaluator):
    # Component implementation
```

### Agent Creation
```python
from flock.core import Flock, FlockFactory

flock = Flock(model="openai/gpt-4o")
agent = FlockFactory.create_default_agent(
    name="my_agent",
    input="query: str",
    output="result: str"
)
flock.add_agent(agent)
result = flock.run(start_agent="my_agent", input={"query": "test"})
```

### Serialization
- All core classes inherit from `Serializable`
- Support JSON, YAML, and Python dict formats
- Use `to_dict()` / `from_dict()` for persistence

## Development Guidelines

### When Making Changes
1. **Always run diagnostics**: Use `get_diagnostics` tool on modified files
2. **Test serialization**: Many components need to serialize/deserialize correctly
3. **Check imports**: Circular imports are a known issue
4. **Memory management**: Be careful with global state (registry, context)

### Code Style
- Use Pydantic for all data models
- Prefer `async`/`await` for I/O operations
- Type hints are mandatory
- Error handling should be specific (avoid bare `except`)

### Testing
- Mock external dependencies (LLM calls, file systems)
- Use fixtures for complex setup
- Test both success and failure paths
- Verify serialization roundtrips

## Quick Start for Development

1. **Understand the flow**: `Flock` → `FlockAgent` → `Evaluator` → Result
2. **Start with examples**: Check `examples/01-getting-started/`
3. **Read tests**: `tests/core/test_flock_core.py` shows usage patterns
4. **Use the factory**: `FlockFactory.create_default_agent()` for quick setup
5. **Focus on contracts**: Input/output signatures are key

## Web Interface

The framework includes a FastAPI web application at `src/flock/webapp/` with:
- Agent execution interface
- Configuration management
- File upload/download
- Real-time execution monitoring

Start with: `flock.serve()` method on any Flock instance.

## External Dependencies

- **DSPy**: LLM interaction and prompt management
- **Temporal.io**: Workflow orchestration and resilience
- **FastAPI**: Web interface
- **Pydantic**: Data validation and serialization
- **OpenTelemetry**: Observability and tracing

## Next Priority Areas

Based on the review, focus on:
1. **Fixing logging conflicts** in test suite
2. **Improving error handling** patterns
3. **Adding security guidelines** for component development
4. **Thread safety** for registry operations

This should give you a solid foundation to understand and contribute to the Flock framework efficiently!
