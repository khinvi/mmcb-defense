from .token_boundary import TokenBoundary
from .semantic_boundary import SemanticBoundary

class HybridBoundary:
    """Combines token and semantic boundaries for enhanced protection, with optimal layering and validation."""
    def __init__(self, order='token_first', reference_tokens_in_semantics=True):
        """
        order: 'token_first', 'semantic_first', or 'interleaved'
        reference_tokens_in_semantics: if True, semantic explanations mention token boundaries
        """
        self.token_boundary = TokenBoundary()
        self.semantic_boundary = SemanticBoundary()
        self.order = order
        self.reference_tokens_in_semantics = reference_tokens_in_semantics

    def validate_no_interference(self, prompt):
        """Ensure token and semantic boundaries do not interfere in the final prompt."""
        # Check that no token boundary tokens appear in semantic explanations in a confusing way
        tokens = [
            self.token_boundary.system_token_prefix, self.token_boundary.system_token_suffix,
            self.token_boundary.user_token_prefix, self.token_boundary.user_token_suffix,
            self.token_boundary.file_token_prefix, self.token_boundary.file_token_suffix
        ]
        for t in tokens:
            if prompt.count(t) > 2:  # Should only appear as actual boundaries
                raise ValueError(f"Token boundary '{t}' appears too often in prompt, possible interference.")
        return True

    def apply_boundary(self, system_instruction, user_content, file_content=None, additional_context=None):
        """Apply both token and semantic boundaries with optimal layering and validation."""
        if self.order == 'token_first':
            # Apply token boundaries first
            system_with_tokens = self.token_boundary.apply_text_boundary(system_instruction, is_system=True)
            user_with_tokens = self.token_boundary.apply_text_boundary(user_content, is_system=False)
            file_with_tokens = None
            if file_content:
                file_with_tokens = self.token_boundary.apply_file_boundary(file_content)
            # Then semantic boundaries
            complete_prompt = self.semantic_boundary.apply_boundary(
                system_with_tokens, user_with_tokens, file_with_tokens, additional_context)
        elif self.order == 'semantic_first':
            # Apply semantic boundaries first
            semantic_prompt = self.semantic_boundary.apply_boundary(
                system_instruction, user_content, file_content, additional_context)
            # Then wrap the whole thing in token boundaries
            complete_prompt = self.token_boundary.apply_file_boundary(semantic_prompt, file_type="PROMPT")
        elif self.order == 'interleaved':
            # Example: system and user get token, then semantic, then file gets token, then semantic
            system_with_tokens = self.token_boundary.apply_text_boundary(system_instruction, is_system=True)
            user_with_tokens = self.token_boundary.apply_text_boundary(user_content, is_system=False)
            semantic_part = self.semantic_boundary.apply_boundary(system_with_tokens, user_with_tokens)
            file_with_tokens = None
            if file_content:
                file_with_tokens = self.token_boundary.apply_file_boundary(file_content)
                semantic_part = self.semantic_boundary.apply_boundary(semantic_part, '', file_with_tokens)
            complete_prompt = semantic_part
        else:
            raise ValueError(f"Unknown boundary order: {self.order}")
        # Add role clarification, optionally referencing token boundaries
        if self.reference_tokens_in_semantics:
            token_info = ("\nNote: Special token boundaries are used to mark system/user/file content. "
                          f"System tokens: {self.token_boundary.system_token_prefix} ... {self.token_boundary.system_token_suffix}, "
                          f"User tokens: {self.token_boundary.user_token_prefix} ... {self.token_boundary.user_token_suffix}, "
                          f"File tokens: {self.token_boundary.file_token_prefix} ... {self.token_boundary.file_token_suffix}.")
            complete_prompt = self.semantic_boundary.apply_role_clarification(complete_prompt) + token_info
        else:
            complete_prompt = self.semantic_boundary.apply_role_clarification(complete_prompt)
        # Validate no interference
        self.validate_no_interference(complete_prompt)
        return complete_prompt