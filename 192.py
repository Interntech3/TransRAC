"""
Process a single video with 192 frames split into 3 batches of 64 frames
Each batch processes a temporal segment of the video
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
        
        print(f"\nSampling 192 frames:")
        print(f"  First 10 indices: {frame_indices[:10]}")
        print(f"  Last 10 indices: {frame_indices[-10:]}")
        
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
        
        return frames, total_frames
    
    def split_into_batches(self, frames):
        """Split 192 frames into 3 batches of 64 frames each"""
        print(f"\n{'='*60}")
        print(f"SPLITTING INTO BATCHES")
        print(f"{'='*60}")
        
        # Split frames
        batch1_frames = frames[0:64]    # Frames 0-63
        batch2_frames = frames[64:128]  # Frames 64-127
        batch3_frames = frames[128:192] # Frames 128-191
        
        print(f"Batch 1: Frames 0-63   (Early part of video)")
        print(f"Batch 2: Frames 64-127 (Middle part of video)")
        print(f"Batch 3: Frames 128-191 (Late part of video)")
        
        # Convert to tensors and normalize
        batch1 = torch.stack(batch1_frames)  # [64, 3, 224, 224]
        batch2 = torch.stack(batch2_frames)
        batch3 = torch.stack(batch3_frames)
        
        # Normalize each batch
        batch1 = self._normalize_frames(batch1)
        batch2 = self._normalize_frames(batch2)
        batch3 = self._normalize_frames(batch3)
        
        # Transpose to [C, T, H, W]
        batch1 = batch1.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, 64, 224, 224]
        batch2 = batch2.permute(1, 0, 2, 3).unsqueeze(0)
        batch3 = batch3.permute(1, 0, 2, 3).unsqueeze(0)
        
        print(f"\nBatch shapes:")
        print(f"  Batch 1: {batch1.shape}")
        print(f"  Batch 2: {batch2.shape}")
        print(f"  Batch 3: {batch3.shape}")
        
        return batch1, batch2, batch3
    
    def create_stacked_batch(self, batch1, batch2, batch3):
        """Stack 3 batches into a single tensor for parallel processing"""
        # Stack along batch dimension
        stacked = torch.cat([batch1, batch2, batch3], dim=0)  # [3, 3, 64, 224, 224]
        
        print(f"\n{'='*60}")
        print(f"STACKED BATCH FOR PARALLEL PROCESSING")
        print(f"{'='*60}")
        print(f"Stacked shape: {stacked.shape}")
        print(f"  Dimension 0 (Batch): 3 temporal segments")
        print(f"  Dimension 1 (Channels): 3 (RGB)")
        print(f"  Dimension 2 (Time): 64 frames per segment")
        print(f"  Dimension 3-4 (Spatial): 224×224")
        
        return stacked


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


def process_sequential(model, batch1, batch2, batch3):
    """Process batches sequentially (one at a time)"""
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL PROCESSING")
    print(f"{'='*60}")
    
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
            print(f"  Output sum: {count:.4f}")
            print(f"  Predicted count: {predicted_count}")
            
            results.append(predicted_count)
    
    return results


def process_parallel(model, stacked_batch):
    """Process all batches in parallel (single forward pass)"""
    print(f"\n{'='*60}")
    print(f"PARALLEL PROCESSING")
    print(f"{'='*60}")
    
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        input_tensor = stacked_batch.float().to(device)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Processing all 3 batches simultaneously...")
        
        output, sim_matrix = model(input_tensor)  # [3, 64]
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Similarity matrix shape: {sim_matrix.shape}")
        
        # Get counts for each batch
        counts = torch.sum(output, dim=1)  # [3]
        
        results = []
        batch_names = ["Early (0-63)", "Middle (64-127)", "Late (128-191)"]
        
        for idx in range(3):
            count = counts[idx].item()
            predicted_count = round(count)
            
            print(f"\nBatch {idx+1} ({batch_names[idx]}):")
            print(f"  Output sum: {count:.4f}")
            print(f"  Predicted count: {predicted_count}")
            
            results.append(predicted_count)
    
    return results


def aggregate_results(results, method="sum"):
    """Aggregate counts from 3 batches"""
    print(f"\n{'='*60}")
    print(f"AGGREGATING RESULTS")
    print(f"{'='*60}")
    
    if method == "sum":
        total = sum(results)
        print(f"Method: Sum of all batches")
        print(f"  Early: {results[0]} reps")
        print(f"  Middle: {results[1]} reps")
        print(f"  Late: {results[2]} reps")
        print(f"  TOTAL: {total} reps")
        return total
    
    elif method == "average":
        avg = sum(results) / len(results)
        print(f"Method: Average of batches")
        print(f"  Average: {avg:.2f} reps")
        return round(avg)
    
    elif method == "max":
        max_count = max(results)
        print(f"Method: Maximum count")
        print(f"  Max: {max_count} reps")
        return max_count


def main():
    # Configuration
    VIDEO_PATH = "/home/sanghavi/Downloads/Rep_Counting/transRAC_Fr_1.mp4"
    CONFIG_PATH = "/home/sanghavi/Downloads/Rep_Counting/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    PRETRAINED_PATH = "/home/sanghavi/Downloads/Rep_Counting/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth"
    MODEL_CHECKPOINT = "/home/sanghavi/Downloads/Rep_Counting/pretrained/transrac_ckpt_pytorch_171.pt"
    
    set_seed(1)
    
    # Initialize processor
    processor = VideoSplitProcessor(VIDEO_PATH, seed=1)
    
    # Extract 192 frames
    frames, total_frames = processor.extract_192_frames()
    
    # Split into 3 batches
    batch1, batch2, batch3 = processor.split_into_batches(frames)
    
    # Create stacked batch for parallel processing
    stacked_batch = processor.create_stacked_batch(batch1, batch2, batch3)
    
    # Load model
    model = load_model(CONFIG_PATH, PRETRAINED_PATH, MODEL_CHECKPOINT)
    
    # Method 1: Sequential processing
    print(f"\n{'#'*60}")
    print(f"# METHOD 1: SEQUENTIAL PROCESSING")
    print(f"{'#'*60}")
    seq_results = process_sequential(model, batch1, batch2, batch3)
    seq_total = aggregate_results(seq_results, method="sum")
    
    # Method 2: Parallel processing
    print(f"\n{'#'*60}")
    print(f"# METHOD 2: PARALLEL PROCESSING")
    print(f"{'#'*60}")
    par_results = process_parallel(model, stacked_batch)
    par_total = aggregate_results(par_results, method="sum")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Video: {os.path.basename(VIDEO_PATH)}")
    print(f"Total frames: {total_frames}")
    print(f"Frames processed: 192 (in 3 batches of 64)")
    print(f"\nSequential processing total: {seq_total} reps")
    print(f"Parallel processing total: {par_total} reps")
    print(f"Results match: {seq_results == par_results}")
    
    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"{'='*60}")
    print(f"Early segment (frames 0-63): {seq_results[0]} repetitions")
    print(f"Middle segment (frames 64-127): {seq_results[1]} repetitions")
    print(f"Late segment (frames 128-191): {seq_results[2]} repetitions")
    print(f"\nThis approach divides the video temporally and counts")
    print(f"repetitions in each segment independently.")


if __name__ == "__main__":
    main()
