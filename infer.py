"""
Comprehensive preprocessing verification script
Tests byte-for-byte parity between inference and training preprocessing
"""
 
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import sys
import random
 
# Insert paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Video-Swin-Transformer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TransRAC"))
from TransRAC.models.TransRAC import TransferModel
 
def set_deterministic_mode(seed=1):
    """Set deterministic mode - using seed=1 like test_looping.py"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set deterministic mode with seed: {seed}")
 
class PreprocessingVerifier:
    """Verifies preprocessing matches training exactly"""
    
    def __init__(self, video_path, num_frames=64, seed=1):
        self.video_path = video_path
        self.num_frames = num_frames
        self.seed = seed
        set_deterministic_mode(seed)
    
    def extract_frames_cv2_method(self):
        """Extract frames using cv2.resize (INTER_LINEAR) - alternative method"""
        np.random.seed(self.seed)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video: {total_frames} frames, {fps} FPS")
        
        # Frame sampling - verify exact indices
        if total_frames <= self.num_frames:
            frame_indices = list(range(total_frames))
            frame_indices.extend([total_frames-1] * (self.num_frames - total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        
        frame_indices = sorted(set(frame_indices))
        print(f"Sampled frame indices (first 10): {frame_indices[:10]}")
        print(f"Sampled frame indices (last 10): {frame_indices[-10:]}")
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frame_indices:
                # CRITICAL: Verify BGR->RGB conversion happens exactly once
                print(f"Frame {frame_count}: Original shape {frame.shape}, dtype {frame.dtype}")
                print(f"  BGR pixel at (100,100): {frame[100,100] if frame.shape[0] > 100 and frame.shape[1] > 100 else 'N/A'}")
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print(f"  RGB pixel at (100,100): {frame_rgb[100,100] if frame_rgb.shape[0] > 100 and frame_rgb.shape[1] > 100 else 'N/A'}")
                
                # Resize using cv2 with INTER_LINEAR (same as PIL's default)
                frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                
                # Convert to tensor manually (instead of using transforms)
                frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1)  # HWC -> CHW
                frame_tensor = frame_tensor / 255.0  # [0, 1]
                
                frames.append(frame_tensor)
                
                # Debug first frame extensively
                if len(frames) == 1:
                    print(f"  Resized shape: {frame_resized.shape}")
                    print(f"  Tensor shape: {frame_tensor.shape}")
                    print(f"  Tensor range: [{frame_tensor.min():.6f}, {frame_tensor.max():.6f}]")
                    print(f"  Tensor mean: {frame_tensor.mean():.6f}")
            
            frame_count += 1
        
        cap.release()
        
        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        frames = frames[:self.num_frames]
        
        # Stack and verify normalization exactly like RepCountA_Loader.py
        video_tensor = torch.stack(frames)  # [T, C, H, W]
        
        # Convert to [0, 255] range
        video_tensor = video_tensor * 255.0
        print(f"\nAfter *255: range [{video_tensor.min():.6f}, {video_tensor.max():.6f}]")
        
        # Apply exact normalization from RepCountA_Loader.py
        video_tensor = video_tensor - 127.5  # frames -= 127.5
        print(f"After -127.5: range [{video_tensor.min():.6f}, {video_tensor.max():.6f}]")
        
        video_tensor = video_tensor / 127.5   # frames /= 127.5
        print(f"After /127.5: range [{video_tensor.min():.6f}, {video_tensor.max():.6f}]")
        
        # Transpose to [C, T, H, W] format
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor.unsqueeze(0), total_frames  # Add batch dimension
    
    def extract_frames_transforms_method(self):
        """Extract frames using torchvision transforms - current method"""
        np.random.seed(self.seed)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= self.num_frames:
            frame_indices = list(range(total_frames))
            frame_indices.extend([total_frames-1] * (self.num_frames - total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        
        frame_indices = sorted(set(frame_indices))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frame_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = transform(frame_rgb)  # Already [0,1] from ToTensor()
                frames.append(frame_tensor)
            
            frame_count += 1
        
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        frames = frames[:self.num_frames]
        
        video_tensor = torch.stack(frames)  # [T, C, H, W]
        
        # Apply normalization to match RepCountA_Loader.py
        video_tensor = video_tensor * 255.0 - 127.5
        video_tensor = video_tensor / 127.5
        
        # Transpose to [C, T, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor.unsqueeze(0), total_frames
    
    def verify_preprocessing_fingerprint(self, tensor, method_name):
        """Print preprocessing fingerprint for comparison"""
        print(f"\n=== {method_name.upper()} FINGERPRINT ===")
        print(f"Final tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Mean: {tensor.mean():.6f}")
        print(f"Std: {tensor.std():.6f}")
        print(f"Min: {tensor.min():.6f}")
        print(f"Max: {tensor.max():.6f}")
        
        # Sample specific pixel values at fixed locations
        b, c, t, h, w = tensor.shape
        test_coords = [
            (0, 0, 0, 0, 0),      # First pixel, first frame
            (0, 1, 0, 112, 112),  # Green channel, center pixel, first frame
            (0, 2, 31, 50, 150),  # Blue channel, mid frame, arbitrary pixel
            (0, 0, 63, 223, 223), # Last pixel, last frame
        ]
        
        for coord in test_coords:
            try:
                value = tensor[coord]
                print(f"Pixel at {coord}: {value:.6f}")
            except IndexError:
                print(f"Pixel at {coord}: Index out of bounds")
        
        return tensor
    
    def compare_methods(self):
        """Compare cv2 vs transforms preprocessing"""
        print("="*60)
        print("PREPROCESSING COMPARISON")
        print("="*60)
        
        tensor_cv2, frames1 = self.extract_frames_cv2_method()
        self.verify_preprocessing_fingerprint(tensor_cv2, "CV2 Method")
        
        print("\n" + "-"*50 + "\n")
        
        tensor_transforms, frames2 = self.extract_frames_transforms_method()
        self.verify_preprocessing_fingerprint(tensor_transforms, "Transforms Method")
        
        # Compare tensors
        print(f"\n=== COMPARISON ===")
        print(f"Tensors equal: {torch.allclose(tensor_cv2, tensor_transforms, atol=1e-6)}")
        print(f"Max difference: {(tensor_cv2 - tensor_transforms).abs().max():.8f}")
        print(f"Mean difference: {(tensor_cv2 - tensor_transforms).abs().mean():.8f}")
        
        return tensor_cv2, tensor_transforms
 
def verify_model_setup(config_path, pretrained_path, model_checkpoint_path):
    """Verify model configuration matches test_looping.py exactly"""
    print("\n" + "="*60)
    print("MODEL SETUP VERIFICATION")
    print("="*60)
    
    # Initialize model exactly like test_looping.py
    model = TransferModel(
        config=config_path,
        checkpoint=pretrained_path,
        num_frames=64,
        scales=[1, 4, 8],
        OPEN=False
    )
    
    print(f"Model OPEN parameter: False (backbone frozen)")
    print(f"Model scales: [1, 4, 8]")
    
    # Wrap with DataParallel like test_looping.py
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model.to(device), device_ids=[0])
    
    # Load checkpoint with detailed logging
    if os.path.exists(model_checkpoint_path):
        print(f"Loading checkpoint: {model_checkpoint_path}")
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print(f"First 5 missing keys: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"First 5 unexpected keys: {unexpected_keys[:5]}")
        
        del checkpoint
    
    return model
 
def test_inference_with_fingerprints(model, tensor, video_path):
    """Test inference with detailed fingerprints"""
    print(f"\n" + "="*60)
    print("INFERENCE WITH FINGERPRINTS")
    print("="*60)
    
    # Pre-inference checks
    input_tensor = tensor.type(torch.FloatTensor).to(next(model.parameters()).device)
    
    print(f"Input to model:")
    print(f"  Shape: {input_tensor.shape}")
    print(f"  Dtype: {input_tensor.dtype}")
    print(f"  Device: {input_tensor.device}")
    print(f"  Mean: {input_tensor.mean():.6f}")
    print(f"  Std: {input_tensor.std():.6f}")
    print(f"  Range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
    
    # Inference
    model.eval()
    with torch.no_grad():
        print(f"\nModel training mode: {model.training}")
        
        output, similarity_matrix = model(input_tensor)
        predicted_count = torch.sum(output, dim=1).round().item()
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Output sum: {torch.sum(output, dim=1).item():.6f}")
        print(f"Predicted count (rounded): {predicted_count}")
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    return predicted_count
 
def main():
    # Configuration
    VIDEO_PATH = "/home/sanghavi/Downloads/Rep_Counting/transRAC_Fr_1.mp4"
    CONFIG_PATH = "/home/sanghavi/Downloads/Rep_Counting/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    PRETRAINED_PATH = "/home/sanghavi/Downloads/Rep_Counting/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth"
    MODEL_CHECKPOINT = "/home/sanghavi/Downloads/Rep_Counting/pretrained/transrac_ckpt_pytorch_171.pt"
    
    # Set seed=1 like test_looping.py
    set_deterministic_mode(1)
    
    # Step 1: Compare preprocessing methods
    verifier = PreprocessingVerifier(VIDEO_PATH, num_frames=64, seed=1)
    tensor_cv2, tensor_transforms = verifier.compare_methods()
    
    # Step 2: Verify model setup
    model = verify_model_setup(CONFIG_PATH, PRETRAINED_PATH, MODEL_CHECKPOINT)
    
    # Step 3: Test inference with both preprocessing methods
    print(f"\nTesting CV2 method:")
    pred1 = test_inference_with_fingerprints(model, tensor_cv2, VIDEO_PATH)
    
    print(f"\nTesting Transforms method:")
    pred2 = test_inference_with_fingerprints(model, tensor_transforms, VIDEO_PATH)
    
    print(f"\n" + "="*60)
    print(f"FINAL RESULTS")
    print(f"="*60)
    print(f"CV2 method prediction: {pred1}")
    print(f"Transforms method prediction: {pred2}")
    print(f"Predictions match: {pred1 == pred2}")
 
if __name__ == "__main__":
    main()