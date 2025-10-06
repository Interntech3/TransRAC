"""
Process video with adaptive aggregation to handle double-counting
Uses 192 frames split into 3 batches with smart counting
Supports real-time camera input and saves captured video
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VideoSplitProcessor:
    """Process video by splitting into temporal segments"""
    
    def __init__(self, seed=1):
        self.seed = seed
        set_seed(seed)
    
    def _normalize_frames(self, frames_tensor):
        frames_tensor = frames_tensor * 255.0
        frames_tensor = (frames_tensor - 127.5) / 127.5
        return frames_tensor
    
    def extract_192_frames_realtime(self, cam_source=0, save_path="captured_video.mp4"):
        """
        Capture 192 frames from a live camera (webcam or RTSP)
        Saves captured video to save_path
        """
        cap = cv2.VideoCapture(cam_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera/stream: {cam_source}")

        print("\n==============================")
        print("  REAL-TIME CAMERA CAPTURE")
        print("==============================")
        print("Capturing 192 frames...")

        frames = []
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (224, 224))  # 30 FPS, 224x224

        while len(frames) < 192:
            ret, frame = cap.read()
            if not ret:
                print("Camera stream dropped / no frame received.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb)
            frames.append(frame_tensor)

            # Save resized frame to video
            frame_resized = cv2.resize(frame, (224, 224))
            out.write(frame_resized)

        cap.release()
        out.release()

        # Pad if fewer than 192
        while len(frames) < 192:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))

        frames = frames[:192]
        print(f"Captured {len(frames)} frames and saved to {save_path}")
        return frames, None, None
    
    def split_into_batches(self, frames):
        """Split 192 frames into 3 batches of 64 frames each"""
        batch1 = torch.stack(frames[0:64])
        batch2 = torch.stack(frames[64:128])
        batch3 = torch.stack(frames[128:192])
        
        batch1 = self._normalize_frames(batch1).permute(1, 0, 2, 3).unsqueeze(0)
        batch2 = self._normalize_frames(batch2).permute(1, 0, 2, 3).unsqueeze(0)
        batch3 = self._normalize_frames(batch3).permute(1, 0, 2, 3).unsqueeze(0)
        
        return batch1, batch2, batch3


def load_model(config_path, pretrained_path, checkpoint_path):
    model = TransferModel(
        config=config_path,
        checkpoint=pretrained_path,
        num_frames=64,
        scales=[1, 4, 8],
        OPEN=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model.to(device), device_ids=[0])
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint
    
    return model


def process_batches(model, batch1, batch2, batch3):
    device = next(model.parameters()).device
    model.eval()
    results = []
    batch_names = ["Early (0-63)", "Middle (64-127)", "Late (128-191)"]
    
    with torch.no_grad():
        for idx, batch in enumerate([batch1, batch2, batch3]):
            input_tensor = batch.float().to(device)
            output, sim_matrix = model(input_tensor)
            count = torch.sum(output, dim=1).item()
            results.append(round(count))
            print(f"\nBatch {idx+1} ({batch_names[idx]}): Predicted {results[-1]} reps")
    
    return results


def adaptive_aggregate(results):
    avg_count = sum(results) / len(results)
    variance = sum((r - avg_count)**2 for r in results) / len(results)
    print(f"\nBatch results: {results}, Average: {avg_count:.2f}, Variance: {variance:.2f}")
    
    if variance < 2.0 and avg_count > 3.5:
        adjusted = sum(results) // 2
        print(f"Double-counting detected, applying correction: {sum(results)} → {adjusted}")
        return adjusted
    else:
        total = sum(results)
        print(f"Normal pattern, final count: {total}")
        return total


def main():
    CONFIG_PATH = "/home/sanghavi/Downloads/Rep_Counting/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    PRETRAINED_PATH = "/home/sanghavi/Downloads/Rep_Counting/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth"
    MODEL_CHECKPOINT = "/home/sanghavi/Downloads/Rep_Counting/pretrained/transrac_ckpt_pytorch_171.pt"
    
    set_seed(1)
    
    processor = VideoSplitProcessor(seed=1)
    
    # Capture live video and save
    frames, _, _ = processor.extract_192_frames_realtime(cam_source=0, save_path="captured_video.mp4")
    
    batch1, batch2, batch3 = processor.split_into_batches(frames)
    model = load_model(CONFIG_PATH, PRETRAINED_PATH, MODEL_CHECKPOINT)
    results = process_batches(model, batch1, batch2, batch3)
    final_count = adaptive_aggregate(results)
    
    print(f"\n🎯 Total repetitions: {final_count}")


if __name__ == "__main__":
    main()
