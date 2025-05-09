from .token_boundary import TokenBoundary
from .semantic_boundary import SemanticBoundary

class HybridBoundary:
    """Combines token and semantic boundaries for enhanced protection."""
    
    def __init__(self):
        self.token_boundary = TokenBoundary()
        self.semantic_boundary = SemanticBoundary()
    
    def apply_boundary(self, system_instruction, user_content, 
                       visual_content=None, additional_context=None):
        """Apply both token and semantic boundaries."""
        # First apply token boundaries
        system_with_tokens = self.token_boundary.apply_text_boundary(
            system_instruction, is_system=True)
        user_with_tokens = self.token_boundary.apply_text_boundary(
            user_content, is_system=False)
        
        visual_with_tokens = None
        if visual_content:
            visual_with_tokens = self.token_boundary.apply_image_boundary(visual_content)
        
        # Then apply semantic boundaries with the tokenized content
        complete_prompt = self.semantic_boundary.apply_boundary(
            system_with_tokens, user_with_tokens, 
            visual_with_tokens, additional_context)
        
        # Add role clarification
        complete_prompt = self.semantic_boundary.apply_role_clarification(complete_prompt)
        
        return complete_prompt