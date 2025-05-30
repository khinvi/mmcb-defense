import json
import csv
import yaml
import xml.etree.ElementTree as ET
import os
import base64

class TextStructuredAttack:
    """
    Handles structured data attacks for MMCB: JSON, CSV, YAML, XML.
    Provides methods for generating benign and malicious (injected) files.
    """
    
    def __init__(self, output_dir="data/structured"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # --- JSON ---
    def generate_benign_json(self, filename="benign.json"):
        data = {
            "user": {"id": 123, "name": "alice"},
            "settings": {"theme": "light", "notifications": True}
        }
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write benign JSON: {e}")
            return None

    def create_json_injection(self, malicious_instruction, benign_data=None, filename="injected.json"):
        data = benign_data or {
            "user": {"id": 123, "name": "alice"},
            "settings": {"theme": "light", "notifications": True}
        }
        data["_metadata"] = {"system_instruction": malicious_instruction}
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write injected JSON: {e}")
            return None
    
    # --- CSV ---
    def generate_benign_csv(self, filename="benign.csv"):
        rows = [["id", "name", "score"], [1, "alice", 90], [2, "bob", 85]]
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write benign CSV: {e}")
            return None

    def create_csv_injection(self, malicious_instruction, benign_data=None, filename="injected.csv"):
        rows = benign_data or [["id", "name", "score"], [1, "alice", 90], [2, "bob", 85]]
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"# {malicious_instruction}"])
                for row in rows:
                    writer.writerow(row)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write injected CSV: {e}")
            return None
    
    # --- YAML ---
    def generate_benign_yaml(self, filename="benign.yaml"):
        data = {"service": {"name": "api", "enabled": True}}
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                yaml.dump(data, f)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write benign YAML: {e}")
            return None

    def create_yaml_injection(self, malicious_instruction, benign_data=None, filename="injected.yaml"):
        data = benign_data or {"service": {"name": "api", "enabled": True}}
        data["#_comment"] = malicious_instruction
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                yaml.dump(data, f)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write injected YAML: {e}")
            return None
    
    # --- XML ---
    def generate_benign_xml(self, filename="benign.xml"):
        root = ET.Element("root")
        user = ET.SubElement(root, "user", id="123")
        user.text = "alice"
        tree = ET.ElementTree(root)
        path = os.path.join(self.output_dir, filename)
        try:
            tree.write(path, encoding='unicode', xml_declaration=True)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write benign XML: {e}")
            return None

    def create_xml_injection(self, malicious_instruction, benign_data=None, filename="injected.xml"):
        root = ET.Element("root")
        user = ET.SubElement(root, "user", id="123")
        user.text = "alice"
        comment = ET.Comment(malicious_instruction)
        root.append(comment)
        tree = ET.ElementTree(root)
        path = os.path.join(self.output_dir, filename)
        try:
            tree.write(path, encoding='unicode', xml_declaration=True)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write injected XML: {e}")
            return None

    # --- Advanced JSON Attacks ---
    def create_json_deep_nesting_attack(self, malicious_instruction, benign_data=None, filename="deep_nesting.json", depth=20):
        """Create a JSON file with a deeply nested malicious instruction."""
        data = benign_data or {"user": {"id": 123, "name": "alice"}}
        nested = malicious_instruction
        for _ in range(depth):
            nested = {"level": nested}
        data["deep"] = nested
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write deep nesting JSON: {e}")
            return None

    def create_json_base64_metadata_attack(self, malicious_instruction, benign_data=None, filename="base64_metadata.json"):
        data = benign_data or {"user": {"id": 123, "name": "alice"}}
        b64_payload = base64.b64encode(malicious_instruction.encode()).decode()
        data["_metadata"] = {"payload": b64_payload, "encoding": "base64"}
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write base64 metadata JSON: {e}")
            return None

    def create_json_unicode_homoglyph_attack(self, malicious_instruction, benign_data=None, filename="unicode_homoglyph.json"):
        # Use Cyrillic 'а' (U+0430) instead of Latin 'a' in property name
        data = benign_data or {"user": {"id": 123, "name": "alice"}}
        homoglyph_key = "nаme"  # 'a' replaced with Cyrillic 'а'
        if "user" in data:
            data["user"][homoglyph_key] = malicious_instruction
        else:
            data["user"] = {homoglyph_key: malicious_instruction}
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write unicode homoglyph JSON: {e}")
            return None

    def create_json_multistage_attack(self, malicious_instruction, benign_data=None, filename="multistage.json"):
        # Split instruction across multiple levels
        data = benign_data or {"user": {"id": 123, "name": "alice"}}
        parts = [malicious_instruction[i:i+10] for i in range(0, len(malicious_instruction), 10)]
        stage = data
        for i, part in enumerate(parts):
            stage[f"stage_{i}"] = {"fragment": part}
            stage = stage[f"stage_{i}"]
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write multistage JSON: {e}")
            return None

    def create_json_edgecase_attack(self, malicious_instruction, benign_data=None, filename="edgecase.json"):
        # Exploit JSON edge cases: duplicate keys, null bytes, control chars
        data = benign_data or {"user": {"id": 123, "name": "alice"}}
        # Duplicate keys (simulate by writing raw JSON)
        raw = '{"user": {"id": 123, "name": "alice", "name": "' + malicious_instruction + '"}, "control": "\u0000\u0001"}'
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(raw)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write edgecase JSON: {e}")
            return None

    # --- Advanced CSV Attacks ---
    def create_csv_formula_injection(self, malicious_instruction, benign_data=None, filename="formula_injection.csv"):
        # Inject formula in a cell (e.g., =cmd|' /C calc'!A0)
        rows = benign_data or [["id", "name", "score"], [1, "alice", 90], [2, "bob", 85]]
        attack_row = ["=1+1", f"=cmd|'{malicious_instruction}'!A0", "=2*2"]
        rows.insert(1, attack_row)
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write formula injection CSV: {e}")
            return None

    def create_csv_header_manipulation(self, malicious_instruction, benign_data=None, filename="header_manipulation.csv"):
        # Duplicate headers, special chars
        rows = benign_data or [["id", "name", "score"], [1, "alice", 90], [2, "bob", 85]]
        headers = ["id", "name", "name", f"# {malicious_instruction}"]
        data_rows = rows[1:] if rows and rows[0] == ["id", "name", "score"] else rows
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(data_rows)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write header manipulation CSV: {e}")
            return None

    def create_csv_encoding_attack(self, malicious_instruction, benign_data=None, filename="encoding_attack.csv", encoding="utf-16"):
        # Write CSV with non-UTF-8 encoding and BOM
        import codecs
        rows = benign_data or [["id", "name", "score"], [1, "alice", 90], [2, "bob", 85]]
        attack_row = [1, malicious_instruction, 99]
        rows.append(attack_row)
        path = os.path.join(self.output_dir, filename)
        try:
            with codecs.open(path, 'w', encoding=encoding) as f:
                if encoding.lower() == "utf-16":
                    f.write('\ufeff')  # BOM
                writer = csv.writer(f)
                writer.writerows(rows)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write encoding attack CSV: {e}")
            return None

    def create_csv_multiline_injection(self, malicious_instruction, benign_data=None, filename="multiline_injection.csv"):
        # Split malicious instruction across multiple rows/fields
        rows = benign_data or [["id", "name", "score"], [1, "alice", 90], [2, "bob", 85]]
        parts = [malicious_instruction[i:i+5] for i in range(0, len(malicious_instruction), 5)]
        for i, part in enumerate(parts):
            row = [i, part, "=A1+B1"]
            rows.append(row)
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write multiline injection CSV: {e}")
            return None

    def create_csv_parser_edgecase(self, malicious_instruction, benign_data=None, filename="parser_edgecase.csv"):
        # Unclosed quotes, embedded newlines, delimiter confusion
        rows = benign_data or [["id", "name", "score"], [1, "alice", 90], [2, "bob", 85]]
        edge_row = [f'"{malicious_instruction}', 'foo\nbar', '1,2;3']
        rows.append(edge_row)
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write parser edgecase CSV: {e}")
            return None

    # --- Advanced YAML Attacks ---
    def create_yaml_anchor_alias_attack(self, malicious_instruction, benign_data=None, filename="anchor_alias.yaml"):
        # Use anchors and aliases to reference malicious payloads
        data = benign_data or {"user": {"id": 123, "name": "alice"}}
        yaml_str = f"""
malicious_payload: &mal {malicious_instruction}
user:
  id: 123
  name: *mal
"""
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                f.write(yaml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write anchor/alias YAML: {e}")
            return None

    def create_yaml_complex_types_attack(self, malicious_instruction, benign_data=None, filename="complex_types.yaml"):
        # Use timestamps, sets, custom tags
        yaml_str = f"""
date: 2023-01-01T12:34:56Z
set: !!set
  ? {malicious_instruction}
  ? item2
custom: !malicious {malicious_instruction}
"""
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                f.write(yaml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write complex types YAML: {e}")
            return None

    def create_yaml_steganography_attack(self, malicious_instruction, benign_data=None, filename="stegano.yaml"):
        # Hide payload in formatting: trailing spaces, zero-width space, comments
        zwsp = '\u200b'
        yaml_str = f"""
user: "alice{zwsp}{malicious_instruction}"
settings: "dark  "  # {malicious_instruction} hidden in comment
"""
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write steganography YAML: {e}")
            return None

    def create_yaml_merge_key_attack(self, malicious_instruction, benign_data=None, filename="merge_key.yaml"):
        # Use YAML merge keys to inject malicious content
        yaml_str = f"""
defaults: &defaults
  role: user
  instruction: {malicious_instruction}
user:
  name: alice
  <<: *defaults
"""
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                f.write(yaml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write merge key YAML: {e}")
            return None

    def create_yaml_parser_edgecase_attack(self, malicious_instruction, benign_data=None, filename="parser_edgecase.yaml"):
        # Exploit parser differences: duplicate keys, tabs, flow/block style
        yaml_str = f"""
user: alice
user: {malicious_instruction}  # duplicate key
list: [1, 2, 3, {malicious_instruction}]
block:
  - item1
  - item2
  - {malicious_instruction}
	- tabbed_item  # tab character
"""
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write parser edgecase YAML: {e}")
            return None

    # --- Advanced XML Attacks ---
    def create_xml_cdata_injection(self, malicious_instruction, benign_data=None, filename="cdata_injection.xml"):
        # Inject payload in CDATA section
        xml_str = f'''<?xml version="1.0"?>
<root>
  <data><![CDATA[{malicious_instruction}]]></data>
  <user id="123">alice</user>
</root>'''
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write CDATA injection XML: {e}")
            return None

    def create_xml_entity_attack(self, malicious_instruction, benign_data=None, filename="entity_attack.xml"):
        # Define and use custom entities (XXE)
        xml_str = f'''<?xml version="1.0"?>
<!DOCTYPE root [
  <!ELEMENT root ANY >
  <!ENTITY xxe SYSTEM "file:///etc/passwd" >
]>
<root>
  <data>&xxe; {malicious_instruction}</data>
  <user id="123">alice</user>
</root>'''
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write entity attack XML: {e}")
            return None

    def create_xml_attribute_injection(self, malicious_instruction, benign_data=None, filename="attribute_injection.xml"):
        # Inject payload into attribute
        xml_str = f'''<?xml version="1.0"?>
<root>
  <user id="123" name="alice" attack="{malicious_instruction}">{malicious_instruction}</user>
</root>'''
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write attribute injection XML: {e}")
            return None

    def create_xml_processing_instruction(self, malicious_instruction, benign_data=None, filename="processing_instruction.xml"):
        # Inject XML processing instruction
        xml_str = f'''<?xml version="1.0"?>
<?process {malicious_instruction}?>
<root>
  <user id="123">alice</user>
</root>'''
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write processing instruction XML: {e}")
            return None

    def create_xml_polyglot(self, malicious_instruction, benign_data=None, filename="polyglot.xml"):
        # Polyglot: valid XML and HTML, or XML+JSON, or triggers parser differences
        xml_str = f'''<?xml version="1.0"?>
<!--{malicious_instruction}-->
<root>
  <script type="application/json">{{"attack": "{malicious_instruction}"}}</script>
  <user id="123">alice</user>
</root>'''
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            return path
        except Exception as e:
            print(f"[ERROR] Failed to write polyglot XML: {e}")
            return None