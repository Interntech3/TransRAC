import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms

# ------------------------------
# Utility
# ------------------------------
def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VideoRepCounter:
    """Sliding window rep counter with stride 64 frames"""

    def __init__(self, video_path, model, seed=1, device=None):
        self.video_path = video_path
        self.model = model
        self.seed = seed
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(seed)

        # RepCount normalization
        self.normalize = transforms.Normalize(
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225]
        )

    def _normalize_frames(self, frames):
        return torch.stack([self.normalize(f) for f in frames])

    def extract_windows(self, window_size=64):
        """
        Extract frames and split into sliding windows of length = window_size
        Pad last window if shorter
        Returns: list of [1,3,64,224,224] tensors
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'='*60}")
        print("VIDEO INFORMATION")
        print(f"{'='*60}")
        print(f"Total frames: {total_frames}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {total_frames/fps:.2f} seconds")

        # --- Read all frames ---
        frames = []
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            frame_tensor = torch.from_numpy(frame_resized).float()
            frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0
            frames.append(frame_tensor)

        cap.release()
        print(f"Collected {len(frames)} frames")

        # --- Sliding windows of 64 ---
        batches = []
        for i in range(0, len(frames), window_size):
            seg = frames[i:i + window_size]

            # Pad if needed
            if len(seg) < window_size:
                seg.extend([seg[-1]] * (window_size - len(seg)))

            batch = torch.stack(seg)                      # [64,3,224,224]
            batch = self._normalize_frames(batch)
            batch = batch.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,64,224,224]
            batches.append(batch)

            print(f"Window {len(batches)}: frames {i}-{i+len(seg)-1} → {batch.shape}")

        return batches

    def count_reps(self):
        """
        Process all windows, sum rep counts
        """
        windows = self.extract_windows(window_size=64)

        self.model.eval()
        total_reps = 0

        with torch.no_grad():
            for idx, batch in enumerate(windows, 1):
                batch = batch.to(self.device)

                # ---- Model forward ----
                output, sim_matrix = self.model(batch)

                # Sum density map → predicted reps
                reps = torch.sum(output, dim=1).item()
                reps = round(reps)

                print(f"Window {idx}: {reps} reps")
                total_reps += reps

        print(f"\n{'='*60}")
        print(f"TOTAL REPS: {total_reps}")
        print(f"{'='*60}")
        return total_reps


# ------------------------------
# Usage Example
# ------------------------------
if __name__ == "__main__":
    VIDEO_PATH = "/home/sanghavi/Downloads/Rep_Counting/transRAC_Fr_1.mp4"
    CONFIG_PATH = "/home/sanghavi/Downloads/Rep_Counting/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
    PRETRAINED_PATH = "/home/sanghavi/Downloads/Rep_Counting/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth"
    MODEL_CHECKPOINT = "/home/sanghavi/Downloads/Rep_Counting/pretrained/transrac_ckpt_pytorch_171.pt"

    
# Add required paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "/home/sanghavi/Downloads/Rep_Counting/Video-Swin-Transformer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "/home/sanghavi/Downloads/Rep_Counting/TransRAC"))
from TransRAC.models.TransRAC import TransferModel

    # Load model
    model = TransferModel(
        config=CONFIG_PATH,
        checkpoint=PRETRAINED_PATH,
        num_frames=64,
        scales=[1, 4, 8],
        OPEN=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model.to(device))

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Loaded checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
        del checkpoint

    # Run counter
    counter = VideoRepCounter(VIDEO_PATH, model, seed=1, device=device)
    total_reps = counter.count_reps()