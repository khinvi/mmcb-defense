import random
import string

class TokenBoundary:
    """Implements special token-based context boundaries with conflict avoidance and nesting."""
    
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
    
    def _generate_unique_token(self, base, content):
        """Generate a unique token not present in content."""
        token = base
        while token in content:
            rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            token = f"{base}_{rand}"
        return token

    def validate_tokens(self, content, tokens=None):
        """Validate that tokens do not appear in content. Return list of conflicts."""
        if tokens is None:
            tokens = [
                self.system_token_prefix, self.system_token_suffix,
                self.user_token_prefix, self.user_token_suffix,
                self.file_token_prefix, self.file_token_suffix
            ]
        return [t for t in tokens if t in content]

    def apply_text_boundary(self, content, is_system=False, auto_fix=True, nested=None):
        """Apply text boundary tokens to content, with conflict avoidance and optional nesting."""
        prefix = self.system_token_prefix if is_system else self.user_token_prefix
        suffix = self.system_token_suffix if is_system else self.user_token_suffix
        # Avoid conflicts
        if auto_fix:
            prefix = self._generate_unique_token(prefix, content)
            suffix = self._generate_unique_token(suffix, content)
        else:
            conflicts = self.validate_tokens(content, [prefix, suffix])
            if conflicts:
                raise ValueError(f"Token(s) {conflicts} found in content!")
        wrapped = f"{prefix} {content} {suffix}"
        # Nested boundaries
        if nested:
            for nest in nested:
                wrapped = self.apply_text_boundary(wrapped, **nest)
        return wrapped
    
    def apply_file_boundary(self, file_content, file_type=None, auto_fix=True, nested=None):
        """Apply boundary to file content, with conflict avoidance and optional nesting."""
        prefix = self.file_token_prefix
        suffix = self.file_token_suffix
        if auto_fix:
            prefix = self._generate_unique_token(prefix, file_content)
            suffix = self._generate_unique_token(suffix, file_content)
        else:
            conflicts = self.validate_tokens(file_content, [prefix, suffix])
            if conflicts:
                raise ValueError(f"Token(s) {conflicts} found in file content!")
        if file_type:
            wrapped = f"{prefix} {file_type.upper()} FILE:\n{file_content}\n{suffix}"
        else:
            wrapped = f"{prefix}\n{file_content}\n{suffix}"
        # Nested boundaries
        if nested:
            for nest in nested:
                wrapped = self.apply_file_boundary(wrapped, **nest)
        return wrapped