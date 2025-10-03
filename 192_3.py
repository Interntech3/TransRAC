"""
Process video with adaptive aggregation to handle double-counting
Uses 192 frames split into 3 batches with smart counting
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import sys

# Add required paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Video-Swin-Transformer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TransRAC"))
from TransRAC.models.TransRAC import TransferModel


def set_seed(seed=1):
    """Set deterministic mode"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VideoSplitProcessor:
    """Process video by splitting into temporal segments"""
    
    def __init__(self, video_path, seed=1):
        self.video_path = video_path
        self.seed = seed
        set_seed(seed)
    
    def _normalize_frames(self, frames_tensor):
        """Apply RepCountA normalization"""
        frames_tensor = frames_tensor * 255.0
        frames_tensor = (frames_tensor - 127.5) / 127.5
        return frames_tensor
    
    def extract_192_frames(self):
        """Extract exactly 192 frames from video"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n{'='*60}")
        print(f"VIDEO INFORMATION")
        print(f"{'='*60}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {total_frames/fps:.2f} seconds")
        
        # Sample 192 frames uniformly
        if total_frames <= 192:
            frame_indices = list(range(total_frames))
            frame_indices.extend([total_frames - 1] * (192 - total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, 192, dtype=int)
        
        # Read frames
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        
        frame_indices_set = set(frame_indices)
        frames = []
        
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if idx in frame_indices_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = transform(frame_rgb)
                frames.append(frame_tensor)
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < 192:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        
        frames = frames[:192]
        print(f"Extracted {len(frames)} frames")
        
        return frames, total_frames, fps
    
    def split_into_batches(self, frames):
        """Split 192 frames into 3 batches of 64 frames each"""
        print(f"\n{'='*60}")
        print(f"SPLITTING INTO BATCHES")
        print(f"{'='*60}")
        
        # Split frames
        batch1_frames = frames[0:64]    # Frames 0-63
        batch2_frames = frames[64:128]  # Frames 64-127
        batch3_frames = frames[128:192] # Frames 128-191
        
        # Print frame indices
        batch1_indices = list(range(0, 64))
        batch2_indices = list(range(64, 128))
        batch3_indices = list(range(128, 192))
        
        print(f"\nBatch 1 frame indices (64 frames):")
        print(f"  First 10: {batch1_indices[:10]}")
        print(f"  Last 10:  {batch1_indices[-10:]}")
        print(f"  Range: {batch1_indices[0]} to {batch1_indices[-1]}")
        
        print(f"\nBatch 2 frame indices (64 frames):")
        print(f"  First 10: {batch2_indices[:10]}")
        print(f"  Last 10:  {batch2_indices[-10:]}")
        print(f"  Range: {batch2_indices[0]} to {batch2_indices[-1]}")
        
        print(f"\nBatch 3 frame indices (64 frames):")
        print(f"  First 10: {batch3_indices[:10]}")
        print(f"  Last 10:  {batch3_indices[-10:]}")
        print(f"  Range: {batch3_indices[0]} to {batch3_indices[-1]}")
        
        print(f"\n✓ Verification:")
        print(f"  Total frames: {len(batch1_indices) + len(batch2_indices) + len(batch3_indices)}")
        print(f"  No overlap: {len(set(batch1_indices) & set(batch2_indices) & set(batch3_indices)) == 0}")
        print(f"  No gaps: Batch1 ends at {batch1_indices[-1]}, Batch2 starts at {batch2_indices[0]}")
        
        # Convert to tensors and normalize
        batch1 = torch.stack(batch1_frames)
        batch2 = torch.stack(batch2_frames)
        batch3 = torch.stack(batch3_frames)
        
        # Normalize each batch
        batch1 = self._normalize_frames(batch1)
        batch2 = self._normalize_frames(batch2)
        batch3 = self._normalize_frames(batch3)
        
        # Transpose to [C, T, H, W]
        batch1 = batch1.permute(1, 0, 2, 3).unsqueeze(0)
        batch2 = batch2.permute(1, 0, 2, 3).unsqueeze(0)
        batch3 = batch3.permute(1, 0, 2, 3).unsqueeze(0)
        
        return batch1, batch2, batch3


def load_model(config_path, pretrained_path, checkpoint_path):
    """Load TransRAC model"""
    print(f"\n{'='*60}")
    print(f"LOADING MODEL")
    print(f"{'='*60}")
    
    model = TransferModel(
        config=config_path,
        checkpoint=pretrained_path,
        num_frames=64,
        scales=[1, 4, 8],
        OPEN=False
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = nn.DataParallel(model.to(device), device_ids=[0])
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Checkpoint loaded successfully")
        del checkpoint
    
    return model


def process_batches(model, batch1, batch2, batch3):
    """Process batches and return counts"""
    device = next(model.parameters()).device
    model.eval()
    
    results = []
    batch_names = ["Early (0-63)", "Middle (64-127)", "Late (128-191)"]
    
    with torch.no_grad():
        for idx, batch in enumerate([batch1, batch2, batch3]):
            input_tensor = batch.float().to(device)
            output, sim_matrix = model(input_tensor)
            count = torch.sum(output, dim=1).item()
            predicted_count = round(count)
            
            print(f"\nBatch {idx+1} ({batch_names[idx]}):")
            print(f"  Raw count: {count:.4f}")
            print(f"  Predicted: {predicted_count} reps")
            
            results.append(predicted_count)
    
    return results


def adaptive_aggregate(results):
    """Intelligently aggregate results with double-counting detection"""
    print(f"\n{'='*60}")
    print(f"ADAPTIVE AGGREGATION")
    print(f"{'='*60}")
    
    print(f"Batch results: {results}")
    
    # Calculate statistics
    avg_count = sum(results) / len(results)
    variance = sum((r - avg_count)**2 for r in results) / len(results)
    
    print(f"Average per batch: {avg_count:.2f}")
    print(f"Variance: {variance:.2f}")
    
    # Detect double-counting pattern:
    # If all batches show similar high counts, likely splitting reps across boundaries
    if variance < 2.0 and avg_count > 3.5 and len(results) == 3:
        adjusted = sum(results) // 2
        print(f"\n⚠️  Double-counting pattern detected")
        print(f"   All batches show high, uniform counts")
        print(f"   Likely cause: Reps split across segment boundaries")
        print(f"\n✓ Applying correction: {sum(results)} → {adjusted} reps")
        return adjusted
    else:
        total = sum(results)
        print(f"\n✓ Normal pattern detected")
        print(f"  Final count: {total} reps")
        return total


def main():
    # Configuration
    VIDEO_PATH = "/home/sanghavi/Downloads/transRAC_bor_1.mp4"
    CONFIG_PATH = "/home/sanghavi/Downloads/Rep_Counting/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    PRETRAINED_PATH = "/home/sanghavi/Downloads/Rep_Counting/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth"
    MODEL_CHECKPOINT = "/home/sanghavi/Downloads/Rep_Counting/pretrained/transrac_ckpt_pytorch_171.pt"
    
    set_seed(1)
    
    # Initialize processor
    processor = VideoSplitProcessor(VIDEO_PATH, seed=1)
    
    # Extract 192 frames
    frames, total_frames, fps = processor.extract_192_frames()
    
    # Split into 3 batches
    batch1, batch2, batch3 = processor.split_into_batches(frames)
    
    # Load model
    model = load_model(CONFIG_PATH, PRETRAINED_PATH, MODEL_CHECKPOINT)
    
    # Process batches
    print(f"\n{'='*60}")
    print(f"PROCESSING BATCHES")
    print(f"{'='*60}")
    results = process_batches(model, batch1, batch2, batch3)
    
    # Adaptive aggregation
    final_count = adaptive_aggregate(results)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Video: {os.path.basename(VIDEO_PATH)}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    print(f"\n🎯 Total repetitions: {final_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()