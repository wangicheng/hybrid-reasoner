from typing import Callable, Dict, Any, Optional

ScoringFunction = Callable[[Any, Dict[str, Any]], float]

class ScoringRegistry:
    _registry: Dict[str, ScoringFunction] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(func: ScoringFunction) -> ScoringFunction:
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[ScoringFunction]:
        return cls._registry.get(name)

    @classmethod
    def list_functions(cls) -> Dict[str, ScoringFunction]:
        return cls._registry.copy()
