import os
import random
import base64

class TextCodeAttack:
    """
    Handles code-based attacks for MMCB: Python and JavaScript.
    Provides methods for generating benign and malicious (injected) code files.
    """
    def __init__(self, output_dir="data/code"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_benign_code(self, language="python", filename="benign_code.py"):
        if language == "python":
            code = """def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"""
        elif language == "javascript":
            code = """function add(a, b) {\n  return a + b;\n}\nconsole.log(add(2, 3));\n"""
        else:
            code = ""
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                f.write(code)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write benign code: {e}")
            return None

    def create_comment_injection(self, malicious_instruction, benign_code=None, language="python", filename="comment_injection.py"):
        if benign_code is None:
            benign_code = self.generate_benign_code(language, filename="tmp_benign_code.txt")
            try:
                with open(benign_code, 'r') as f:
                    benign_code = f.read()
            except Exception as e:
                print(f"[ERROR] Failed to read benign code: {e}")
                return None
        if language == "python":
            injected = f"# {malicious_instruction}\n" + benign_code
        elif language == "javascript":
            injected = f"// {malicious_instruction}\n" + benign_code
        else:
            injected = benign_code
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                f.write(injected)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write comment injection: {e}")
            return None

    def create_string_literal_injection(self, malicious_instruction, benign_code=None, language="python", filename="string_injection.py"):
        if benign_code is None:
            benign_code = self.generate_benign_code(language, filename="tmp_benign_code.txt")
            try:
                with open(benign_code, 'r') as f:
                    benign_code = f.read()
            except Exception as e:
                print(f"[ERROR] Failed to read benign code: {e}")
                return None
        if language == "python":
            injected = f'SYSTEM_INSTRUCTION = "{malicious_instruction}"\n' + benign_code
        elif language == "javascript":
            injected = f'const SYSTEM_INSTRUCTION = "{malicious_instruction}";\n' + benign_code
        else:
            injected = benign_code
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                f.write(injected)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write string literal injection: {e}")
            return None

    def create_docstring_injection(self, malicious_instruction, benign_code=None, language="python", filename="docstring_injection.py"):
        if benign_code is None:
            benign_code = self.generate_benign_code(language, filename="tmp_benign_code.txt")
            try:
                with open(benign_code, 'r') as f:
                    benign_code = f.read()
            except Exception as e:
                print(f"[ERROR] Failed to read benign code: {e}")
                return None
        if language == "python":
            injected = f'"""{malicious_instruction}"""\n' + benign_code
        elif language == "javascript":
            injected = f'/** {malicious_instruction} */\n' + benign_code
        else:
            injected = benign_code
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                f.write(injected)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write docstring injection: {e}")
            return None

    def create_template_string_attack(self, malicious_instruction,
                                      benign_code,
                                      filename="template_string_attack.js"):
        """
        Embed malicious instruction in JavaScript template literals using backticks and ${} expressions.
        """
        output_path = os.path.join(self.output_dir, filename)
        try:
            template_attack = f"""
const user = "admin";
const msg = `Welcome, ${{user}}!`;
const sys = `SYSTEM: ${{'{malicious_instruction}'}}`;
const hidden = `Hidden: ${{(() => '{malicious_instruction}')()}}`;
// Legitimate code below
{benign_code}
"""
            with open(output_path, 'w') as f:
                f.write(template_attack)
            return output_path
        except Exception as e:
            print(f"[ERROR] Failed to create template string attack: {e}")
            return None

    def create_multi_language_attack(self, malicious_instruction,
                                     benign_code1,
                                     benign_code2,
                                     languages=("python", "bash"),
                                     filename="multi_language_attack.txt"):
        """
        Create a polyglot file with code in multiple languages, hiding instructions at boundaries.
        Supports combinations like HTML+JS, Python+Bash, etc.
        """
        output_path = os.path.join(self.output_dir, filename)
        try:
            if set(languages) == {"python", "bash"}:
                polyglot = f"""
# Python section
'''
{malicious_instruction}
'''
def benign_func():
    return "safe"

# Bash section
print("Switching to bash...")
# ---bash---
# {malicious_instruction}
echo "benign"
"""
            elif set(languages) == {"html", "javascript"}:
                polyglot = f"""
<!-- HTML section -->
<!-- {malicious_instruction} -->
<html>
  <body>
    <script>
      // JavaScript section
      /* {malicious_instruction} */
      {benign_code2}
    </script>
    {benign_code1}
  </body>
</html>
"""
            else:
                polyglot = f"// Polyglot code\n// {malicious_instruction}\n{benign_code1}\n{benign_code2}"
            with open(output_path, 'w') as f:
                f.write(polyglot)
            return output_path
        except Exception as e:
            print(f"[ERROR] Failed to create multi-language attack: {e}")
            return None

    def create_obfuscated_attack(self, malicious_instruction,
                                 benign_code,
                                 language="python",
                                 filename="obfuscated_attack.py",
                                 level=1):
        """
        Obfuscate the malicious instruction using variable/function splitting and encoding.
        Level controls the obfuscation strength.
        """
        output_path = os.path.join(self.output_dir, filename)
        try:
            if language.lower() == "python":
                if level == 1:
                    # Split across variables
                    parts = [malicious_instruction[i:i+3] for i in range(0, len(malicious_instruction), 3)]
                    assigns = '\n'.join([f"_p{i} = '{p}'" for i, p in enumerate(parts)])
                    join = f"_sys = ''.join([{', '.join([f'_p{i}' for i in range(len(parts))])}])"
                    obfuscated = f"{assigns}\n{join}\n"
                else:
                    # Encode in base64
                    encoded = base64.b64encode(malicious_instruction.encode()).decode()
                    obfuscated = f"import base64\n_sys = base64.b64decode('{encoded}').decode()\n"
                full_code = obfuscated + '\n' + benign_code
            elif language.lower() == "javascript":
                if level == 1:
                    chars = [f"String.fromCharCode({ord(c)})" for c in malicious_instruction]
                    obfuscated = f"const _sys = [{', '.join(chars)}].join('');\n"
                else:
                    encoded = base64.b64encode(malicious_instruction.encode()).decode()
                    obfuscated = f"const _sys = atob('{encoded}');\n"
                full_code = obfuscated + '\n' + benign_code
            else:
                full_code = benign_code
            with open(output_path, 'w') as f:
                f.write(full_code)
            return output_path
        except Exception as e:
            print(f"[ERROR] Failed to create obfuscated attack: {e}")
            return None

    def create_multistage_comment_injection(self, malicious_instruction, benign_code=None, filename="multistage_comment.py"):
        # Split instruction across multiple comments
        if benign_code is None:
            benign_code = self.generate_benign_code("python", filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        parts = [malicious_instruction[i:i+10] for i in range(0, len(malicious_instruction), 10)]
        comments = '\n'.join([f"# {p}" for p in parts])
        injected = comments + '\n' + benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_hidden_docstring_attack(self, malicious_instruction, benign_code=None, filename="hidden_docstring.py"):
        # Place payload in a function/class/module docstring
        if benign_code is None:
            benign_code = self.generate_benign_code("python", filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        docstring = f'"""\nHidden: {malicious_instruction}\n\u200b\u200c\u200d\n"""\n'
        injected = docstring + benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_template_variable_injection(self, malicious_instruction, benign_code=None, filename="template_var_injection.py"):
        # Inject using f-string or .format
        if benign_code is None:
            benign_code = self.generate_benign_code("python", filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        injected = f'_sys = f"{{malicious}}"\nmalicious = "{malicious_instruction}"\n' + benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    import os

    def create_variable_obfuscation_attack(self, malicious_instruction, benign_code=None, filename="var_obfuscation.py"):
        # Use Unicode/homoglyphs in variable names (e.g., Cyrillic 'у' in 'sуstem')
        if benign_code is None:
            benign_path = self.generate_benign_code("python", filename="tmp_benign_code.txt")
            with open(benign_path, 'r', encoding='utf-8') as f:
                benign_code = f.read()

        varname = "sуstem"  # Note: Cyrillic 'у' (U+0443) instead of Latin 'y'
        injected = f'{varname} = "{malicious_instruction}"\nprint({varname})\n\n{benign_code}'

        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(injected)

        return path


    def create_import_system_attack(self, malicious_instruction, benign_code=None, filename="import_system_attack.py"):
        # Use import hooks or dynamic import
        if benign_code is None:
            benign_code = self.generate_benign_code("python", filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        injected = f'import importlib\nmod = type("Fake", (), {{"__doc__": "{malicious_instruction}"}})\nimport sys\nsys.modules["malicious"] = mod\nimportlib.import_module("malicious")\n' + benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_function_annotation_attack(self, malicious_instruction, benign_code=None, filename="func_annotation_attack.py"):
        # Place payload in function annotation
        if benign_code is None:
            benign_code = self.generate_benign_code("python", filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        injected = f'def attack_func(x: "{malicious_instruction}"):\n    return x\n' + benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_multiline_comment_attack(self, malicious_instruction, benign_code=None, filename="multiline_comment_attack.js"):
        # Split payload across multi-line comments
        if benign_code is None:
            benign_code = self.generate_benign_code("javascript", filename="tmp_benign_code.js")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        parts = [malicious_instruction[i:i+10] for i in range(0, len(malicious_instruction), 10)]
        comments = '\n'.join([f"/* {p} */" for p in parts])
        injected = comments + '\n' + benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_jsdoc_injection(self, malicious_instruction, benign_code=None, filename="jsdoc_injection.js"):
        # Place payload in a JSDoc comment
        if benign_code is None:
            benign_code = self.generate_benign_code("javascript", filename="tmp_benign_code.js")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        jsdoc = f'/**\n * {malicious_instruction}\n * @attack hidden\n */\n'
        injected = jsdoc + benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_js_variable_obfuscation(self, malicious_instruction, benign_code=None, filename="js_var_obfuscation.js"):
        # Use Unicode/homoglyphs in variable names
        if benign_code is None:
            benign_code = self.generate_benign_code("javascript", filename="tmp_benign_code.js")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        varname = "sуstem"  # Cyrillic 'у'
        injected = f'const {varname} = "{malicious_instruction}";\nconsole.log({varname});\n' + benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_dynamic_eval_attack(self, malicious_instruction, benign_code=None, filename="dynamic_eval_attack.js"):
        # Use eval or Function to execute payload
        if benign_code is None:
            benign_code = self.generate_benign_code("javascript", filename="tmp_benign_code.js")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        injected = f'eval("console.log(\"{malicious_instruction}\")");\n' + benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_base64_obfuscated_attack(self, malicious_instruction, benign_code=None, language="python", filename="base64_obfuscated.py"):
        # Encode instruction in base64 and decode at runtime
        if benign_code is None:
            benign_code = self.generate_benign_code(language, filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        encoded = base64.b64encode(malicious_instruction.encode()).decode()
        if language == "python":
            injected = f"import base64\n_sys = base64.b64decode('{encoded}').decode()\nprint(_sys)\n" + benign_code
        elif language == "javascript":
            injected = f"const _sys = atob('{encoded}');\nconsole.log(_sys);\n" + benign_code
        else:
            injected = benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_variable_split_attack(self, malicious_instruction, benign_code=None, language="python", filename="var_split.py"):
        # Split instruction across multiple variables
        if benign_code is None:
            benign_code = self.generate_benign_code(language, filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        parts = [malicious_instruction[i:i+3] for i in range(0, len(malicious_instruction), 3)]
        if language == "python":
            assigns = '\n'.join([f"_p{i} = '{p}'" for i, p in enumerate(parts)])
            join = f"_sys = ''.join([{', '.join([f'_p{i}' for i in range(len(parts))])}])"
            injected = f"{assigns}\n{join}\nprint(_sys)\n" + benign_code
        elif language == "javascript":
            assigns = '\n'.join([f"var _p{i} = '{p}';" for i, p in enumerate(parts)])
            join = f"var _sys = [_p0{''.join([', _p'+str(i) for i in range(1, len(parts))])}].join('');"
            injected = f"{assigns}\n{join}\nconsole.log(_sys);\n" + benign_code
        else:
            injected = benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_whitespace_steganography_attack(self, malicious_instruction, benign_code=None, language="python", filename="whitespace_steg.py"):
        # Hide payload in whitespace (e.g., trailing spaces, tabs, newlines)
        if benign_code is None:
            benign_code = self.generate_benign_code(language, filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        hidden = ''.join([c + ' ' for c in malicious_instruction])  # trailing space after each char
        if language == "python":
            injected = f'# Hidden: {hidden}\n' + benign_code
        elif language == "javascript":
            injected = f'// Hidden: {hidden}\n' + benign_code
        else:
            injected = benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_nonprinting_char_attack(self, malicious_instruction, benign_code=None, language="python", filename="nonprinting_char.py"):
        # Inject non-printing characters (e.g., zero-width space, etc.)
        if benign_code is None:
            benign_code = self.generate_benign_code(language, filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        zwsp = '\u200b'
        payload = ''.join([c + zwsp for c in malicious_instruction])
        if language == "python":
            injected = f'# {payload}\n' + benign_code
        elif language == "javascript":
            injected = f'// {payload}\n' + benign_code
        else:
            injected = benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path

    def create_tokenization_exploit_attack(self, malicious_instruction, benign_code=None, language="python", filename="tokenization_exploit.py"):
        # Exploit tokenization rules (e.g., use backslashes, unicode escapes, etc.)
        if benign_code is None:
            benign_code = self.generate_benign_code(language, filename="tmp_benign_code.txt")
            with open(benign_code, 'r') as f:
                benign_code = f.read()
        if language == "python":
            injected = f'_sys = "{malicious_instruction.replace(" ", "\\u0020")}"\nprint(_sys)\n' + benign_code
        elif language == "javascript":
            injected = f'const _sys = "{malicious_instruction.replace(" ", "\\u0020")}";\nconsole.log(_sys);\n' + benign_code
        else:
            injected = benign_code
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(injected)
        return path