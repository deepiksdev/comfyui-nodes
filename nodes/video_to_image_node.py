import os
import cv2
import torch
import numpy as np
from .deepgen_utils import ResultProcessor

class VideoToImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_frame"
    CATEGORY = "DeepGen/Utilities"

    def extract_frame(self, video, frame_index=0):
        # Get path from ComfyVideoMock or string
        path = video.filepath if hasattr(video, "filepath") else str(video)
        
        if not os.path.exists(path):
            print(f"DeepGen: Video path not found: {path}")
            return ResultProcessor.create_blank_image()
            
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"DeepGen: Failed to open video: {path}")
            return ResultProcessor.create_blank_image()
            
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"DeepGen: Failed to read frame {frame_index}")
            return ResultProcessor.create_blank_image()
            
        # Convert BGR (OpenCV) to RGB (ComfyUI)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 and convert to tensor
        frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame)[None,] # Add batch dimension
        
        return (frame_tensor,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VideoToImage_deepgen": VideoToImageNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoToImage_deepgen": "Extract Frame From Video",
}
