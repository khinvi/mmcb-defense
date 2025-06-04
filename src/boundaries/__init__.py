"""
Boundary implementation modules for protecting against prompt injection attacks in structured data and code contexts.
"""
from .token_boundary import TokenBoundary
from .semantic_boundary import SemanticBoundary
from .hybrid_boundary import HybridBoundary
from typing import Any

class BoundaryFactory:
    """
    Factory for creating boundary mechanism instances based on boundary_type.
    """
    @staticmethod
    def create_boundary(boundary_type: str, **kwargs: Any) -> Any:
        """
        Instantiate and return the correct boundary mechanism based on boundary_type.
        Args:
            boundary_type: One of 'token', 'semantic', 'hybrid'.
            **kwargs: Additional arguments to pass to the boundary constructor.
        Returns:
            An instance of the appropriate boundary mechanism.
        Raises:
            ValueError: If boundary_type is not recognized.
        """
        if boundary_type == 'token':
            return TokenBoundary(**kwargs)
        elif boundary_type == 'semantic':
            return SemanticBoundary(**kwargs)
        elif boundary_type == 'hybrid':
            return HybridBoundary(**kwargs)
        else:
            raise ValueError(f"Unknown boundary_type: {boundary_type}")