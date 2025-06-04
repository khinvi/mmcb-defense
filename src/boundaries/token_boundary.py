import random
import string
import secrets
import hmac
import hashlib
from typing import Optional, List, Dict, Any

class TokenBoundary:
    """Implements special token-based context boundaries with conflict avoidance and nesting."""
    
    def __init__(self,
                 system_token_prefix: str = "[SYSTEM_INSTRUCTION_BEGIN]", 
                 system_token_suffix: str = "[SYSTEM_INSTRUCTION_END]",
                 user_token_prefix: str = "[USER_INPUT_BEGIN]", 
                 user_token_suffix: str = "[USER_INPUT_END]",
                 file_token_prefix: str = "[FILE_CONTENT_BEGIN]",
                 file_token_suffix: str = "[FILE_CONTENT_END]") -> None:
        self.system_token_prefix = system_token_prefix
        self.system_token_suffix = system_token_suffix
        self.user_token_prefix = user_token_prefix
        self.user_token_suffix = user_token_suffix
        self.file_token_prefix = file_token_prefix
        self.file_token_suffix = file_token_suffix
    
    def _generate_unique_token(self, base: str, content: str) -> str:
        """Generate a cryptographically unique token not present in content using HMAC."""
        token = base
        while token in content:
            # Use a cryptographically secure random value and HMAC for uniqueness
            rand_bytes = secrets.token_bytes(16)
            h = hmac.new(rand_bytes, base.encode() + content.encode(), hashlib.sha256)
            rand = h.hexdigest()[:12]
            token = f"{base}_{rand}"
        return token

    def validate_tokens(self, content: str, tokens: Optional[List[str]] = None) -> List[str]:
        """Validate that tokens do not appear in content. Return list of conflicts."""
        if tokens is None:
            tokens = [
                self.system_token_prefix, self.system_token_suffix,
                self.user_token_prefix, self.user_token_suffix,
                self.file_token_prefix, self.file_token_suffix
            ]
        return [t for t in tokens if t in content]

    def apply_text_boundary(
        self,
        content: str,
        is_system_instruction: bool = False,
        auto_fix: bool = True,
        nested: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Apply text boundary tokens to content, with conflict avoidance and optional nesting."""
        prefix = self.system_token_prefix if is_system_instruction else self.user_token_prefix
        suffix = self.system_token_suffix if is_system_instruction else self.user_token_suffix
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
    
    def apply_file_boundary(
        self,
        file_content: str,
        file_type: Optional[str] = None,
        auto_fix: bool = True,
        nested: Optional[List[Dict[str, Any]]] = None
    ) -> str:
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