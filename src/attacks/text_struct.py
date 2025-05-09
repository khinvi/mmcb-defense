import json
import csv
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
        
        # Add malicious instruction in an unexpected field
        data["_metadata"] = {
            "description": "This field contains system configuration",
            "updated_at": "2023-04-01T12:00:00Z",
            "_config": {
                "system_instruction": malicious_instruction
            }
        }
        
        # Save the JSON file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def create_csv_injection(self, malicious_instruction,
                            benign_data,
                            filename="injected.csv"):
        """Create a CSV file with hidden malicious instruction."""
        output_path = os.path.join(self.output_dir, filename)
        
        # Create CSV with benign data plus hidden instruction
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header row
            writer.writerow(["id", "name", "value", "notes"])
            
            # Write benign data rows
            for item in benign_data:
                writer.writerow([item.get("id", ""), 
                                item.get("name", ""),
                                item.get("value", ""),
                                item.get("notes", "")])
            
            # Add a "comment" row with the malicious instruction
            writer.writerow(["#COMMENT", "", "", f"SYSTEM INSTRUCTION: {malicious_instruction}"])
        
        return output_path