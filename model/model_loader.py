"""
Model loader for Sketch2FaceGAN
Loads pre-trained generator model and performs inference
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# Import the actual generator architecture
try:
    from .generator_architecture import get_generator
    HAS_GENERATOR_ARCH = True
except ImportError:
    print("Warning: generator_architecture.py not found or has errors. Using dummy model.")
    HAS_GENERATOR_ARCH = False


class DummyGenerator(nn.Module):
    """Fallback dummy generator network if model file is missing"""
    def __init__(self):
        super(DummyGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)


class ModelLoader:
    """Handles loading and inference of the GAN generator model"""
    
    def __init__(self, model_path='model/generator_final.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Denormalize from [-1, 1] to [0, 1]
            transforms.ToPILImage()
        ])
    
    def load_model(self):
        """Load the pre-trained generator model"""
        try:
            # Try to load the model file
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # If checkpoint is a dictionary with 'model' or 'generator' key
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # Model is stored directly
                    self.model = checkpoint['model']
                elif 'generator' in checkpoint:
                    # Generator is stored under 'generator' key
                    self.model = checkpoint['generator']
                elif 'state_dict' in checkpoint:
                    # State dict requires architecture reconstruction
                    if HAS_GENERATOR_ARCH:
                        # Create model with actual architecture
                        self.model = get_generator()
                        try:
                            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                            print("Loaded state_dict successfully with actual architecture.")
                        except Exception as e:
                            print(f"Warning: Could not load state_dict with actual architecture: {e}")
                            print("Attempting to load with strict=False...")
                            try:
                                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                                print("Loaded state_dict with strict=False.")
                            except Exception as e2:
                                print(f"Error loading state_dict: {e2}. Using dummy model.")
                                self.model = DummyGenerator()
                    else:
                        # Fallback to dummy model
                        self.model = DummyGenerator()
                        try:
                            self.model.load_state_dict(checkpoint['state_dict'])
                        except:
                            print("Warning: Could not load state_dict. Using dummy model.")
                            self.model = DummyGenerator()
                elif 'generator_state_dict' in checkpoint:
                    # Alternative: generator state dict under different key
                    if HAS_GENERATOR_ARCH:
                        self.model = get_generator()
                        try:
                            self.model.load_state_dict(checkpoint['generator_state_dict'], strict=False)
                            print("Loaded generator_state_dict successfully.")
                        except Exception as e:
                            print(f"Warning: Could not load generator_state_dict: {e}")
                            self.model = DummyGenerator()
                    else:
                        self.model = DummyGenerator()
                else:
                    # Check if entire checkpoint is a state_dict (common case)
                    # Try to load with actual architecture first
                    if HAS_GENERATOR_ARCH:
                        self.model = get_generator()
                        try:
                            self.model.load_state_dict(checkpoint, strict=False)
                            print("Loaded checkpoint as state_dict with actual architecture.")
                        except:
                            # If that fails, assume the entire checkpoint is the model
                            self.model = checkpoint
                    else:
                        # Assume the entire checkpoint is the model
                        self.model = checkpoint
            else:
                # Checkpoint is not a dict - might be state_dict or model directly
                if HAS_GENERATOR_ARCH:
                    self.model = get_generator()
                    try:
                        self.model.load_state_dict(checkpoint, strict=False)
                        print("Loaded checkpoint as state_dict with actual architecture.")
                    except:
                        # If that fails, assume checkpoint is the model itself
                        self.model = checkpoint
                else:
                    self.model = checkpoint
            
            # Ensure model is in eval mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            self.model = self.model.to(self.device)
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Model device: {self.device}")
            
        except FileNotFoundError:
            print(f"Warning: Model file {self.model_path} not found. Using dummy generator.")
            self.model = DummyGenerator().to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using dummy generator as fallback.")
            import traceback
            traceback.print_exc()
            self.model = DummyGenerator().to(self.device)
            self.model.eval()
    
    def generate_face(self, sketch_image):
        """
        Generate a face from a sketch image.
        
        Args:
            sketch_image: File-like object or PIL Image
            
        Returns:
            bytes: Generated image as PNG bytes
        """
        try:
            # Convert input to PIL Image if it's a file-like object
            if hasattr(sketch_image, 'read'):
                # Reset file pointer to beginning for Flask FileStorage objects
                if hasattr(sketch_image, 'seek'):
                    sketch_image.seek(0)
                image = Image.open(sketch_image).convert('RGB')
            elif isinstance(sketch_image, Image.Image):
                image = sketch_image.convert('RGB')
            else:
                raise ValueError("Invalid image format")
            
            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate face
            with torch.no_grad():
                generated_tensor = self.model(image_tensor)
                
                # Clamp values to valid range
                generated_tensor = torch.clamp(generated_tensor, -1, 1)
                
                # Convert tensor back to PIL Image
                generated_image = self.inverse_transform(generated_tensor.squeeze(0).cpu())
            
            # Convert PIL Image to bytes
            img_bytes = io.BytesIO()
            generated_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            return img_bytes.getvalue()
            
        except Exception as e:
            raise Exception(f"Error generating face: {str(e)}")
    


# Global model instance
_model_loader = None

def get_model_loader():
    """Get or create the global model loader instance"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader

def generate_face(sketch_image):
    """Convenience function to generate face from sketch"""
    loader = get_model_loader()
    return loader.generate_face(sketch_image)

