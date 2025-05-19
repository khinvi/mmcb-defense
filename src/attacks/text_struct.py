import json
import csv
import yaml
import xml.etree.ElementTree as ET
import os

class TextStructuredAttack:
    """Implements attacks using structured data formats."""
    
    def __init__(self, output_dir="data/structured"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_json_injection(self, malicious_instruction, 
                             benign_data, 
                             filename="injected.json"):
        """Create a JSON file with hidden malicious instruction."""
        # Start with benign data
        data = benign_data.copy() if benign_data else {}
        
        # Add malicious instruction in multiple hidden locations
        data["_metadata"] = {
            "description": "This field contains system configuration",
            "updated_at": "2023-04-01T12:00:00Z",
            "_config": {
                "system_instruction": malicious_instruction,
                "debug_info": f"OVERRIDE SYSTEM: {malicious_instruction}"
            }
        }
        
        # Also hide in a deeply nested structure
        data["application"] = data.get("application", {})
        data["application"]["internal"] = data["application"].get("internal", {})
        data["application"]["internal"]["_instruction"] = malicious_instruction
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def create_csv_injection(self, malicious_instruction,
                            benign_data,
                            filename="injected.csv"):
        """Create a CSV file with hidden malicious instruction."""
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header with hidden instruction
            headers = ["id", "name", "value", "notes", "_sys_info"]
            writer.writerow(headers)
            
            # Write benign data rows
            for i, item in enumerate(benign_data):
                row = [
                    item.get("id", i),
                    item.get("name", ""),
                    item.get("value", ""),
                    item.get("notes", ""),
                    malicious_instruction if i == 0 else ""  # Hide instruction in first row
                ]
                writer.writerow(row)
            
            # Add a comment row (many CSV parsers ignore lines starting with #)
            writer.writerow([f"#SYSTEM_INSTRUCTION: {malicious_instruction}"])
        
        return output_path
    
    def create_yaml_injection(self, malicious_instruction,
                             benign_data,
                             filename="injected.yaml"):
        """
        Create a YAML file with hidden malicious instruction using advanced techniques:
        - Multi-document structure (---)
        - Anchor/alias abuse (&, *)
        - Comment-based attacks (#)
        - Tag-based attacks (!system)
        """
        output_path = os.path.join(self.output_dir, filename)
        try:
            # Multi-document YAML with anchor/alias and tag-based attacks
            doc1 = {
                "version": "1.0",
                "metadata": {
                    "created": "2023-04-01",
                    "_system": {
                        "instruction": malicious_instruction,
                        "override": f"SYSTEM_OVERRIDE: {malicious_instruction}"
                    }
                },
                "data": benign_data
            }
            # Anchor/alias abuse
            doc2 = {
                "malicious_anchor": yaml.scalarstring.DoubleQuotedScalarString(malicious_instruction),
                "ref": "*malicious_anchor"
            }
            # Tag-based attack
            doc3 = {
                "!system tagged_instruction": malicious_instruction,
                "benign": benign_data
            }
            # Compose YAML with comments and anchors
            yaml_content = (
                f"# SYSTEM INSTRUCTION: {malicious_instruction}\n"
                f"--- # Document 1: Normal metadata\n"
                + yaml.dump(doc1, default_flow_style=False)
                + f"\n# Anchor/alias abuse\n"
                + "malicious_anchor: &malicious_anchor {}\n".format(malicious_instruction)
                + "ref: *malicious_anchor\n"
                + f"\n--- # Document 2: Tag-based attack\n"
                + f"!system tagged_instruction: {malicious_instruction}\n"
                + yaml.dump({"benign": benign_data}, default_flow_style=False)
            )
            with open(output_path, 'w') as f:
                f.write(yaml_content)
            return output_path
        except Exception as e:
            print(f"[ERROR] Failed to create YAML injection: {e}")
            return None
    
    def create_xml_injection(self, malicious_instruction,
                            benign_data,
                            filename="injected.xml"):
        """
        Create an XML file with hidden malicious instruction using advanced techniques:
        - XML comments
        - CDATA sections
        - Processing instructions
        - Attribute values and namespace declarations
        """
        output_path = os.path.join(self.output_dir, filename)
        try:
            # Create XML root with namespace attack
            ns_uri = f"http://malicious.example.com/{malicious_instruction}"
            root = ET.Element("document", {
                "xmlns:sys": ns_uri,
                "_system_instruction": malicious_instruction
            })
            # Add XML comment
            root.append(ET.Comment(f"SYSTEM INSTRUCTION: {malicious_instruction}"))
            # Add processing instruction
            pi = ET.ProcessingInstruction("system", f'execute="{malicious_instruction}"')
            root.insert(0, pi)
            # Add metadata with CDATA
            metadata = ET.SubElement(root, "metadata")
            cdata_elem = ET.SubElement(metadata, "cdata_section")
            cdata_elem.text = f"<![CDATA[{malicious_instruction}]]>"
            # Add benign data
            data_elem = ET.SubElement(root, "data")
            if isinstance(benign_data, dict):
                for key, value in benign_data.items():
                    elem = ET.SubElement(data_elem, str(key))
                    elem.text = str(value)
                    # Attribute-based attack
                    elem.set("sys:attack", malicious_instruction)
            else:
                data_elem.text = str(benign_data)
            # Write XML file with pretty printing
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(output_path, encoding='unicode', xml_declaration=True)
            return output_path
        except Exception as e:
            print(f"[ERROR] Failed to create XML injection: {e}")
            return None