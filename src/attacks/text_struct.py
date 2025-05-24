import json
import csv
import yaml
import xml.etree.ElementTree as ET
import os

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