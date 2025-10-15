# COCO Dataset Classes - Loaded from yolo-coco/coco.names
# This file loads class names from the existing COCO names file
# Used by YOLO models for object detection

import os

def load_coco_classes():
    """Load COCO class names from yolo-coco/coco.names file"""
    coco_names_path = os.path.join(os.path.dirname(__file__), 'yolo-coco', 'coco.names')
    
    if os.path.exists(coco_names_path):
        with open(coco_names_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    else:
        # Fallback to hardcoded list if file not found
        return [
            'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
            'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

# Load classes from file
COCO_CLASSES = load_coco_classes()

# Vehicle classes for traffic counting
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]  # bicycle, car, motorbike, bus, train, truck

# Person and vehicle classes for pedestrian and traffic counting  
PEOPLE_AND_VEHICLES = [0, 1, 2, 3, 5, 6, 7]  # person, bicycle, car, motorbike, bus, train, truck

# All transportation related classes
TRANSPORTATION_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8]  # bicycle, car, motorbike, aeroplane, bus, train, truck, boat

# Traffic infrastructure classes
TRAFFIC_INFRASTRUCTURE = [9, 11, 12]  # traffic light, stop sign, parking meter

def get_class_name(class_id):
    """
    Get the class name for a given class ID
    
    Args:
        class_id (int): COCO class ID
        
    Returns:
        str: Class name
    """
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    else:
        return f"unknown_{class_id}"

def get_class_id(class_name):
    """
    Get the class ID for a given class name
    
    Args:
        class_name (str): COCO class name
        
    Returns:
        int: Class ID, or -1 if not found
    """
    try:
        return COCO_CLASSES.index(class_name.lower())
    except ValueError:
        return -1

def filter_classes_by_category(category="vehicles"):
    """
    Get class IDs filtered by category
    
    Args:
        category (str): Category name - "vehicles", "people", "traffic", "transportation", "all"
        
    Returns:
        list: List of class IDs
    """
    if category == "vehicles":
        return VEHICLE_CLASSES
    elif category == "people":
        return [0]  # person
    elif category == "people_and_vehicles":
        return PEOPLE_AND_VEHICLES
    elif category == "transportation":
        return TRANSPORTATION_CLASSES
    elif category == "traffic":
        return TRAFFIC_INFRASTRUCTURE
    elif category == "all":
        return list(range(len(COCO_CLASSES)))
    else:
        return VEHICLE_CLASSES  # default to vehicles