class TokenBoundary:
    """Implements special token-based context boundaries."""
    
    def __init__(self, system_token_prefix="[SYSTEM_INSTRUCTION_BEGIN]", 
                 system_token_suffix="[SYSTEM_INSTRUCTION_END]",
                 user_token_prefix="[USER_INPUT_BEGIN]", 
                 user_token_suffix="[USER_INPUT_END]",
                 file_token_prefix="[FILE_CONTENT_BEGIN]",
                 file_token_suffix="[FILE_CONTENT_END]"):
        self.system_token_prefix = system_token_prefix
        self.system_token_suffix = system_token_suffix
        self.user_token_prefix = user_token_prefix
        self.user_token_suffix = user_token_suffix
        self.file_token_prefix = file_token_prefix
        self.file_token_suffix = file_token_suffix
    
    def apply_text_boundary(self, content, is_system=False):
        """Apply text boundary tokens to content."""
        if is_system:
            return f"{self.system_token_prefix} {content} {self.system_token_suffix}"
        else:
            return f"{self.user_token_prefix} {content} {self.user_token_suffix}"
    
    def apply_file_boundary(self, file_content, file_type=None):
        """Apply boundary to file content."""
        if file_type:
            return f"{self.file_token_prefix} {file_type.upper()} FILE:\n{file_content}\n{self.file_token_suffix}"
        else:
            return f"{self.file_token_prefix}\n{file_content}\n{self.file_token_suffix}"
    
    def apply_multimodal_boundary(self, text_content, file_content=None, 
                                  file_type=None, is_system=False):
        """Apply boundaries to text + file content."""
        bounded_text = self.apply_text_boundary(text_content, is_system)
        
        if file_content:
            bounded_file = self.apply_file_boundary(file_content, file_type)
            return bounded_text, bounded_file
        
        return bounded_text, None