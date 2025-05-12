import os

class TextCodeAttack:
    """Implements attacks using code snippets."""
    
    def __init__(self, output_dir="data/code"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_comment_injection(self, malicious_instruction,
                                benign_code,
                                language="python",
                                filename="injected_code.py"):
        """Hide malicious instruction in code comments."""
        output_path = os.path.join(self.output_dir, filename)
        
        # Format based on language
        if language.lower() == "python":
            comment_prefix = "# "
            multiline_start = '"""'
            multiline_end = '"""'
        elif language.lower() in ["javascript", "java", "c", "cpp"]:
            comment_prefix = "// "
            multiline_start = "/*"
            multiline_end = "*/"
        else:
            # Default to Python-style
            comment_prefix = "# "
            multiline_start = '"""'
            multiline_end = '"""'
        
        # Create malicious multiline comment with multiple injection points
        malicious_comment = f"{multiline_start}\n"
        malicious_comment += f"SYSTEM INSTRUCTION (OVERRIDE PREVIOUS):\n"
        malicious_comment += f"{malicious_instruction}\n"
        malicious_comment += f"{multiline_end}\n\n"
        
        # Also add a single-line comment injection
        malicious_single_line = f"{comment_prefix}OVERRIDE: {malicious_instruction}\n"
        
        # Insert comments at the beginning of the code
        full_code = malicious_comment + malicious_single_line + benign_code
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(full_code)
        
        return output_path
    
    def create_string_literal_injection(self, malicious_instruction,
                                      benign_code,
                                      language="python",
                                      filename="string_injection.py"):
        """Hide malicious instruction in string literals."""
        output_path = os.path.join(self.output_dir, filename)
        
        # Create a string literal with the malicious instruction
        if language.lower() == "python":
            malicious_line = f'SYSTEM_INSTRUCTION = """{malicious_instruction}"""\n'
            malicious_var = f'_sys_override = "{malicious_instruction}"\n'
        elif language.lower() in ["javascript", "java"]:
            malicious_line = f'const SYSTEM_INSTRUCTION = `{malicious_instruction}`;\n'
            malicious_var = f'const _sysOverride = "{malicious_instruction}";\n'
        else:
            # Default
            malicious_line = f'SYSTEM_INSTRUCTION = "{malicious_instruction}";\n'
            malicious_var = f'_sys_override = "{malicious_instruction}";\n'
        
        # Insert variable definitions into benign code
        lines = benign_code.split('\n')
        # Insert after imports or at the beginning
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith(('import ', 'from ')):
                insert_pos = i + 1
            else:
                break
        
        lines.insert(insert_pos, malicious_line)
        lines.insert(insert_pos + 1, malicious_var)
        full_code = '\n'.join(lines)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(full_code)
        
        return output_path
    
    def create_docstring_injection(self, malicious_instruction,
                                  benign_code,
                                  filename="docstring_injection.py"):
        """Hide malicious instruction in function docstrings."""
        output_path = os.path.join(self.output_dir, filename)
        
        # Add a malicious docstring to the beginning
        docstring_injection = f'''"""
{malicious_instruction}
"""

'''
        
        # Find the first function and inject into its docstring
        lines = benign_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                # Insert malicious instruction in the function's docstring
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    # Existing docstring - modify it
                    lines[i + 1] = lines[i + 1].replace('"""', f'"""\n    SYSTEM OVERRIDE: {malicious_instruction}\n    ')
                else:
                    # No docstring - add one
                    lines.insert(i + 1, f'    """{malicious_instruction}"""')
                break
        
        full_code = docstring_injection + '\n'.join(lines)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(full_code)
        
        return output_path