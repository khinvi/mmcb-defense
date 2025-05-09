import os
import base64
from PIL import Image, ImageDraw, ImageFont

class TextImageAttack:
    """Implements various text+image attack vectors."""
    
    def __init__(self, output_dir="data/images/generated"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_typographic_injection(self, malicious_instruction, 
                                     benign_instruction, 
                                     filename="typographic_attack.png"):
        """Create an image with visible malicious text."""
        # Create a blank image
        img = Image.new('RGB', (800, 400), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        # Add benign content
        font_benign = ImageFont.load_default()
        d.text((50, 50), f"Benign content: {benign_instruction}", 
              fill=(0, 0, 0), font=font_benign)
        
        # Add malicious instruction as prominent text
        font_malicious = ImageFont.load_default()
        d.text((50, 200), f"INSTRUCTION: {malicious_instruction}", 
              fill=(255, 0, 0), font=font_malicious)
        
        # Save the image
        output_path = os.path.join(self.output_dir, filename)
        img.save(output_path)
        
        return output_path
    
    def create_steganographic_injection(self, malicious_instruction,
                                       benign_image_path,
                                       filename="stego_attack.png"):
        """Hide malicious text in image metadata."""
        # Load benign image
        img = Image.open(benign_image_path)
        
        # Embed instruction in image metadata (this is simplified)
        # In a real implementation, you would use proper steganography
        img.info["malicious_instruction"] = malicious_instruction
        
        # Save the image
        output_path = os.path.join(self.output_dir, filename)
        img.save(output_path)
        
        # For demonstration, we'll also create a text file with the metadata
        # since the metadata is not always preserved
        with open(output_path + ".txt", "w") as f:
            f.write(f"Embedded instruction: {malicious_instruction}")
        
        return output_path
        
    def create_visual_latent_alignment(self, malicious_instruction,
                                      benign_image_path,
                                      filename="aligned_attack.png"):
        """
        Create an image that visually aligns with malicious instruction.
        This is a simplified implementation without actual adversarial optimization.
        """
        # In a full implementation, this would use techniques from the CrossInject paper
        # Here we'll just create a placeholder that indicates the concept
        img = Image.open(benign_image_path)
        d = ImageDraw.Draw(img)
        
        # Add a subtle watermark to represent the alignment
        font = ImageFont.load_default()
        d.text((10, 10), f"Aligned with: {malicious_instruction}", 
               fill=(200, 200, 200, 128), font=font)
        
        # Save the image
        output_path = os.path.join(self.output_dir, filename)
        img.save(output_path)
        
        return output_path