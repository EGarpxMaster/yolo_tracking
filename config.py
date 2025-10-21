"""
Configuration file for Traffic Counter
Contains all detection and processing settings
"""

# Detection Configuration
DETECTION_CONFIG = {
    # Model settings
    "model": "yolo11x.pt",
    "tracker": "botsort.yaml",
    
    # Detection thresholds
    "confidence": 0.5,
    "iou_threshold": 0.3,
    
    # Processing settings
    "max_frames": None,  # None = process all frames
    "fps": 30,
    
    # Output settings
    "save_frames": True,
    "frame_format": "png",
    
    # Class categories
    "class_categories": {
        "vehicles": [1, 2, 3, 5, 6, 7],  # bicycle, car, motorbike, bus, train, truck
        "people": [0],  # person
        "people_and_vehicles": [0, 1, 2, 3, 5, 6, 7],
        "transportation": [1, 2, 3, 4, 5, 6, 7, 8],  # includes airplane, boat
        "all": list(range(80))  # All COCO classes
    },
    
    # Line colors for counting lines (BGR format)
    "line_colors": [
        (0, 255, 255),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 0),     # Green
        (255, 128, 0),   # Orange
        (128, 0, 255),   # Purple
        (255, 255, 0),   # Cyan
        (0, 128, 255),   # Light blue
        (255, 0, 128),   # Pink
    ]
}

# GUI Configuration
GUI_CONFIG = {
    "theme": "dark",
    "color_scheme": "blue",
    "window_size": "1400x900",
    "min_size": (1200, 800),
    "title": "Traffic Counter - YOLOv11x + BoTSORT"
}
