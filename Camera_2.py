"""
Live 192-frame rep counting at 25 FPS with improved peak detection
Enhanced algorithm to reduce false positives
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import sys
import time
from scipy.signal import find_peaks

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


class VideoProcessor:
    """Capture and process video for 192 frames at 25 FPS"""

    def __init__(self, seed=1):
        set_seed(seed)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def _normalize(self, frames_tensor):
        frames_tensor = frames_tensor * 255.0
        frames_tensor = (frames_tensor - 127.5) / 127.5
        return frames_tensor

    def capture_192_frames(self, cam_source=0, save_path="captured_video.mp4", target_fps=25):
        """Capture 192 frames from webcam with live preview at 25 FPS"""
        cap = cv2.VideoCapture(cam_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera/stream: {cam_source}")

        # Force camera to target FPS (if supported)
        cap.set(cv2.CAP_PROP_FPS, target_fps)

        print("\n==============================")
        print(f"  REAL-TIME CAMERA CAPTURE @ {target_fps} FPS")
        print("==============================")
        print("Capturing 192 frames...")

        frames = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, target_fps, (224, 224))

        start_time = time.time()

        # Calculate ideal frame interval
        frame_interval = 1.0 / target_fps

        while len(frames) < 192:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Camera stream dropped / no frame received.")
                break

            # --- Live preview resized to 800x600 ---
            preview_frame = cv2.resize(frame, (800, 600))
            preview_frame = cv2.putText(
                preview_frame,
                f"Frame {len(frames)+1}/192",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow("📷 Live Capture (press Q to stop)", preview_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped manually.")
                break
            # --------------------

            # Save frame for model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame_rgb)
            frames.append(frame_tensor)

            frame_resized = cv2.resize(frame, (224, 224))
            out.write(frame_resized)

            # Sleep to maintain target FPS
            elapsed = time.time() - frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        end_time = time.time()
        total_time = end_time - start_time

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        while len(frames) < 192:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))

        frames = frames[:192]
        print(f"\n✅ Captured {len(frames)} frames and saved to {save_path}")
        print(f"⏱ Total capture time: {total_time:.2f}s (~{len(frames)/total_time:.2f} FPS)")
        return frames

    def split_batches(self, frames, batch_size=64):
        """Split 192 frames into 3 batches of 64"""
        return [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    def prepare_batch(self, batch):
        """Normalize and prepare for model"""
        batch = torch.stack(batch)
        batch = self._normalize(batch)
        batch = batch.permute(1, 0, 2, 3).unsqueeze(0)
        return batch


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
        print("✅ Checkpoint successfully loaded (64-frame model).")
    else:
        print("⚠️ Checkpoint not found, using model with pretrained backbone only.")

    return model


def count_peaks_improved(output_np, threshold_factor=0.5, min_distance=8, prominence_factor=0.2):
    """
    Improved peak counting with scipy's find_peaks for better accuracy
    
    Args:
        output_np: numpy array of shape [num_frames]
        threshold_factor: multiplier for std to set threshold (increased from 0.3 to 0.5)
        min_distance: minimum frames between peaks (prevents double counting)
        prominence_factor: minimum prominence relative to signal range
    """
    # Convert to float32 for scipy compatibility
    output_np = output_np.astype(np.float32)
    
    # Smooth the signal to reduce noise
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(output_np, sigma=2)
    
    # Calculate adaptive threshold
    threshold = smoothed.mean() + threshold_factor * smoothed.std()
    
    # Calculate prominence requirement
    signal_range = smoothed.max() - smoothed.min()
    min_prominence = prominence_factor * signal_range
    
    # Find peaks with constraints
    peaks, properties = find_peaks(
        smoothed,
        height=threshold,
        distance=min_distance,  # At least 8 frames between peaks (~0.32s at 25 FPS)
        prominence=min_prominence  # Peak must be prominent enough
    )
    
    count = len(peaks)
    print(f"  → Found {count} peaks (threshold={threshold:.3f}, min_distance={min_distance}, prominence={min_prominence:.3f})")
    
    return count


def count_peaks_simple(output_np, threshold_factor=0.5):
    """
    Fallback simple peak counting (if scipy not available)
    Improved with higher threshold
    """
    threshold = output_np.mean() + threshold_factor * output_np.std()
    above_thresh = output_np > threshold

    count = 0
    in_peak = False
    peak_length = 0
    min_peak_length = 3  # Peak must last at least 3 frames
    
    for val in above_thresh:
        if val and not in_peak:
            in_peak = True
            peak_length = 1
        elif val and in_peak:
            peak_length += 1
        elif not val and in_peak:
            if peak_length >= min_peak_length:
                count += 1
            in_peak = False
            peak_length = 0
    
    # Handle last peak
    if in_peak and peak_length >= min_peak_length:
        count += 1
    
    return count


def process_batches(model, batches, threshold_factor=0.5, use_improved=True):
    """
    Run model on each 64-frame batch with improved peak detection
    """
    device = next(model.parameters()).device
    model.eval()
    counts = []

    with torch.no_grad():
        for i, batch in enumerate(batches):
            start_frame = i * 64
            end_frame = start_frame + 63
            input_tensor = batch.float().to(device)
            output, _ = model(input_tensor)

            output_np = output.cpu().numpy().flatten()
            
            # Use improved peak detection if available
            try:
                if use_improved:
                    count = count_peaks_improved(output_np, threshold_factor)
                else:
                    count = count_peaks_simple(output_np, threshold_factor)
            except ImportError:
                print("  ⚠️ scipy not available, using simple peak detection")
                count = count_peaks_simple(output_np, threshold_factor)
            
            counts.append(count)
            print(f"Batch {i+1}: Frames {start_frame:03d}-{end_frame:03d} = {count} reps")

    return counts


def adaptive_aggregate_v2(counts, overlap_penalty=0.8):
    """
    Improved aggregation that's more conservative with overlaps
    
    Args:
        counts: list of rep counts from each batch
        overlap_penalty: reduction factor when batches show similar high counts
    """
    avg_count = sum(counts) / len(counts)
    variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)
    max_count = max(counts)
    min_count = min(counts)
    
    print(f"\n📊 Batch counts: {counts}")
    print(f"   Average: {avg_count:.2f}, Variance: {variance:.2f}, Range: {min_count}-{max_count}")

    # High variance suggests irregular counting - take median or conservative estimate
    if variance > 3.0:
        # Take median of the counts
        sorted_counts = sorted(counts)
        median = sorted_counts[len(sorted_counts) // 2]
        total = median * len(counts)
        print(f"🔄 High variance detected → Using median approach: {median} × {len(counts)} = {total}")
        return round(total)
    
    # Similar high counts suggest overlap - apply penalty
    if avg_count > 3 and variance < 1.5:
        total = sum(counts) * overlap_penalty
        print(f"🔁 Potential overlap detected → Applying penalty: {sum(counts)} × {overlap_penalty} = {total:.1f}")
        return round(total)
    
    # Normal pattern - simple sum
    total = sum(counts)
    print(f"✅ Normal pattern → Total: {total}")
    return total


def main():
    CONFIG_PATH = "/home/sanghavi/Downloads/Rep_Counting/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    PRETRAINED_PATH = "/home/sanghavi/Downloads/Rep_Counting/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth"
    MODEL_CHECKPOINT = "/home/sanghavi/Downloads/Rep_Counting/pretrained/transrac_ckpt_pytorch_171.pt"

    set_seed(1)
    processor = VideoProcessor(seed=1)

    # Capture 192 frames (with live preview at 25 FPS)
    frames = processor.capture_192_frames(cam_source=0, save_path="captured_video.mp4", target_fps=25)

    # Split into 3 batches (64 frames each)
    batches = processor.split_batches(frames, batch_size=64)
    batches = [processor.prepare_batch(b) for b in batches]

    # Load model
    model = load_model(CONFIG_PATH, PRETRAINED_PATH, MODEL_CHECKPOINT)

    # Process batches with improved peak detection
    # Increased threshold_factor from 0.3 to 0.5 for stricter peak detection
    counts = process_batches(model, batches, threshold_factor=0.5, use_improved=True)

    # Improved adaptive aggregation
    final_count = adaptive_aggregate_v2(counts, overlap_penalty=0.8)
    print(f"\n🎯 FINAL COUNT: {final_count} repetitions")


if __name__ == "__main__":
    main()