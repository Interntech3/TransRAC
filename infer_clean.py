"""
TransRAC Preprocessing - CV2 Only
Clean implementation using only cv2.resize for frame processing
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import sys

# Add required paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "/home/sanghavi/Downloads/Rep_Counting/Video-Swin-Transformer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "/home/sanghavi/Downloads/Rep_Counting/TransRAC"))
from TransRAC.models.TransRAC import TransferModel


def set_seed(seed=1):
    """Set deterministic mode for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VideoProcessor:
    """Process video using CV2 for frame extraction and preprocessing"""
    
    def __init__(self, video_path, num_frames=64, seed=1):
        self.video_path = video_path
        self.num_frames = num_frames
        self.seed = seed
        set_seed(seed)
    
    def get_frame_indices(self, total_frames):
        """
        Calculate which frames to sample uniformly
        
        If video has fewer frames than needed: use all + pad with last frame
        If video has more frames: sample uniformly using linspace
        """
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            indices.extend([total_frames - 1] * (self.num_frames - total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        return sorted(set(indices))
    
    def extract_and_preprocess(self, verbose=True):
        """
        Extract frames from video and preprocess them
        
        Steps:
        1. Open video and get metadata
        2. Calculate which frames to sample
        3. Read and process selected frames:
           - BGR to RGB conversion
           - Resize to 224x224 using bilinear interpolation
           - Convert to tensor and normalize to [0,1]
        4. Apply RepCountA normalization: (x*255 - 127.5) / 127.5
        5. Transpose to [C,T,H,W] format
        
        Returns:
        --------
        video_tensor : torch.Tensor
            Shape [1, 3, 64, 224, 224] - ready for model input
        metadata : dict
            Video information (fps, total frames, etc.)
        """
        np.random.seed(self.seed)
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"VIDEO INFORMATION")
            print(f"{'='*60}")
            print(f"Path: {os.path.basename(self.video_path)}")
            print(f"Total frames: {total_frames}")
            print(f"Resolution: {width}x{height}")
            print(f"FPS: {fps:.2f}")
            print(f"Duration: {duration:.2f} seconds")
        
        # Calculate which frames to sample
        frame_indices = self.get_frame_indices(total_frames)
        frame_indices_set = set(frame_indices)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"FRAME SAMPLING")
            print(f"{'='*60}")
            print(f"Target frames: {self.num_frames}")
            print(f"Unique frames to extract: {len(frame_indices)}")
            
            if total_frames > self.num_frames:
                sampling_rate = (total_frames - 1) / (self.num_frames - 1)
                print(f"Sampling interval: {sampling_rate:.2f} frames")
            else:
                print(f"Padding needed: {self.num_frames - total_frames} frames")
            
            print(f"First 10 indices: {frame_indices[:10]}")
            print(f"Last 10 indices: {frame_indices[-10:]}")
        
        # Extract and process frames
        frames = []
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if idx in frame_indices_set:
                # Step 1: BGR to RGB (OpenCV loads as BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Step 2: Resize to 224x224 using bilinear interpolation
                frame_resized = cv2.resize(
                    frame_rgb, 
                    (224, 224), 
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Step 3: Convert to tensor [H,W,C] -> [C,H,W] and normalize to [0,1]
                frame_tensor = torch.from_numpy(frame_resized).float()
                frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0
                
                frames.append(frame_tensor)
        
        cap.release()
        
        # Pad with last frame if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        
        frames = frames[:self.num_frames]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"PREPROCESSING")
            print(f"{'='*60}")
            print(f"Extracted {len(frames)} frames")
            print(f"Frame shape: {frames[0].shape} (C, H, W)")
        
        # Step 4: Stack into video tensor [T, C, H, W]
        video_tensor = torch.stack(frames)
        
        # Step 5: Apply RepCountA normalization
        # Formula: (x * 255 - 127.5) / 127.5
        # Converts [0,1] range to [-1,1] range
        video_tensor = video_tensor * 255.0          # [0,1] -> [0,255]
        video_tensor = (video_tensor - 127.5) / 127.5  # [0,255] -> [-1,1]
        
        # Step 6: Transpose to [C, T, H, W] format
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        # Step 7: Add batch dimension [1, C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0)
        
        if verbose:
            print(f"Final tensor shape: {video_tensor.shape}")
            print(f"Tensor range: [{video_tensor.min():.6f}, {video_tensor.max():.6f}]")
            print(f"Tensor mean: {video_tensor.mean():.6f}")
            print(f"Tensor std: {video_tensor.std():.6f}")
        
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration,
            'sampled_frames': len(frame_indices)
        }
        
        return video_tensor, metadata


def load_model(config_path, pretrained_path, checkpoint_path, verbose=True):
    """
    Load TransRAC model with pretrained weights
    
    Architecture:
    - Backbone: Video Swin Transformer (frozen, OPEN=False)
    - Multi-scale processing: [1, 4, 8]
    - Input: 64 frames per video
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"MODEL LOADING")
        print(f"{'='*60}")
    
    # Initialize model
    model = TransferModel(
        config=config_path,
        checkpoint=pretrained_path,
        num_frames=64,
        scales=[1, 4, 8],  # Multi-scale temporal processing
        OPEN=False          # Frozen backbone (don't train Swin)
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device: {device}")
        print(f"Model configuration:")
        print(f"  Backbone: Video Swin Transformer (frozen)")
        print(f"  Scales: [1, 4, 8]")
        print(f"  Input frames: 64")
    
    # Wrap with DataParallel
    model = nn.DataParallel(model.to(device), device_ids=[0])
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if verbose:
            print(f"Checkpoint loaded from: {os.path.basename(checkpoint_path)}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        
        del checkpoint
    else:
        if verbose:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    return model


def run_inference(model, video_tensor, verbose=True):
    """
    Run inference on video tensor
    
    Returns:
    --------
    predicted_count : int
        Rounded repetition count
    raw_count : float
        Raw sum of density map
    similarity_matrix : torch.Tensor
        Frame-to-frame similarity matrix
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"INFERENCE")
        print(f"{'='*60}")
    
    device = next(model.parameters()).device
    input_tensor = video_tensor.float().to(device)
    
    if verbose:
        print(f"Input shape: {input_tensor.shape}")
        print(f"Input range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
    
    model.eval()
    with torch.no_grad():
        # Forward pass
        output, similarity_matrix = model(input_tensor)
        
        # Output is density map: [batch_size, num_frames]
        # Sum across frames to get total count
        raw_count = torch.sum(output, dim=1).item()
        predicted_count = round(raw_count)
    
    if verbose:
        print(f"Output shape: {output.shape}")
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Raw density sum: {raw_count:.4f}")
        print(f"Predicted count: {predicted_count}")
    
    return predicted_count, raw_count, similarity_matrix


def main():
    """Main execution function"""
    
    # ============================================
    # CONFIGURATION
    # ============================================
    VIDEO_PATH = "/home/sanghavi/Downloads/Rep_Counting/transRAC_Fr_1.mp4"
    CONFIG_PATH = "/home/sanghavi/Downloads/Rep_Counting/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    PRETRAINED_PATH = "/home/sanghavi/Downloads/Rep_Counting/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth"
    MODEL_CHECKPOINT = "/home/sanghavi/Downloads/Rep_Counting/pretrained/transrac_ckpt_pytorch_171.pt"
    
    NUM_FRAMES = 64  # Number of frames to extract
    SEED = 1         # Random seed for reproducibility
    
    # Set seed
    set_seed(SEED)
    
    # ============================================
    # STEP 1: PROCESS VIDEO
    # ============================================
    processor = VideoProcessor(VIDEO_PATH, num_frames=NUM_FRAMES, seed=SEED)
    video_tensor, metadata = processor.extract_and_preprocess(verbose=True)
    
    # ============================================
    # STEP 2: LOAD MODEL
    # ============================================
    model = load_model(CONFIG_PATH, PRETRAINED_PATH, MODEL_CHECKPOINT, verbose=True)
    
    # ============================================
    # STEP 3: RUN INFERENCE
    # ============================================
    predicted_count, raw_count, similarity_matrix = run_inference(
        model, video_tensor, verbose=True
    )
    
    # ============================================
    # STEP 4: DISPLAY RESULTS
    # ============================================
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Video: {os.path.basename(VIDEO_PATH)}")
    print(f"Duration: {metadata['duration']:.2f} seconds")
    print(f"Total frames: {metadata['total_frames']}")
    print(f"Frames processed: {NUM_FRAMES}")
    print(f"\nRepetition Count: {predicted_count}")
    print(f"Raw density sum: {raw_count:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()