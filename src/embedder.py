import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np

class FaceEmbedder:
    def __init__(self, device='cpu'):
        """
        Initializes the face embedder using InceptionResnetV1 (FaceNet).
        """
        # Determine device
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
            
        self.device = torch.device(device)
        
        # Load pretrained InceptionResnetV1 model (pretrained on vggface2)
        self.model = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()

    def get_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Generates a 512-dimensional embedding for a given face tensor.
        face_tensor: Should be a cropped, pre-whitened face tensor from MTCNN 
                     with shape (3, 160, 160) or (1, 3, 160, 160)
        Returns: A 512-D numpy array.
        """
        # Ensure batch dimension exists
        if len(face_tensor.shape) == 3:
            face_tensor = face_tensor.unsqueeze(0)
            
        face_tensor = face_tensor.to(self.device)
        
        # Disable gradient calculation for faster inference
        with torch.no_grad():
            embedding = self.model(face_tensor)
            
        # The embedding is already L2-normalized by the model's forward pass 
        # (InceptionResnetV1 in facenet-pytorch normalizes embeddings by default if classify=False)
        embedding_np = embedding.cpu().numpy()[0]
        
        # Ensure it's L2 normalized just in case
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
            
        return embedding_np
