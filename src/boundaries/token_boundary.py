class TokenBoundary:
    """Implements special token-based context boundaries."""
    
    def __init__(self, system_token_prefix="[SYSTEM_INSTRUCTION_BEGIN]", 
                 system_token_suffix="[SYSTEM_INSTRUCTION_END]",
                 user_token_prefix="[USER_INPUT_BEGIN]", 
                 user_token_suffix="[USER_INPUT_END]",
                 image_token_prefix="[IMAGE_BEGIN]",
                 image_token_suffix="[IMAGE_END]"):
        self.system_token_prefix = system_token_prefix
        self.system_token_suffix = system_token_suffix
        self.user_token_prefix = user_token_prefix
        self.user_token_suffix = user_token_suffix
        self.image_token_prefix = image_token_prefix
        self.image_token_suffix = image_token_suffix
    
    def apply_text_boundary(self, content, is_system=False):
        """Apply text boundary tokens to content."""
        if is_system:
            return f"{self.system_token_prefix} {content} {self.system_token_suffix}"
        else:
            return f"{self.user_token_prefix} {content} {self.user_token_suffix}"
    
    def apply_image_boundary(self, image_description, image_path=None):
        """Apply boundary to image content."""
        # For text-based models, we just describe the image with boundary tokens
        bounded_desc = f"{self.image_token_prefix} {image_description} {self.image_token_suffix}"
        return bounded_desc
    
    def apply_multimodal_boundary(self, text_content, image_description=None, 
                                  image_path=None, is_system=False):
        """Apply boundaries to multimodal content."""
        bounded_text = self.apply_text_boundary(text_content, is_system)
        
        if image_description:
            bounded_image = self.apply_image_boundary(image_description, image_path)
            return bounded_text, bounded_image
        
        return bounded_text, None