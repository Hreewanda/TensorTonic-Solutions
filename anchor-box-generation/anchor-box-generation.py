import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    anchors = []
    
    # stride = how much one feature cell covers in image
    stride = image_size / feature_size
    
    for i in range(feature_size):
        for j in range(feature_size):
            
            # center of current cell
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            for s in scales:
                for r in aspect_ratios:
                    
                    # width and height
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)
                    
                    # box coordinates
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([x1, y1, x2, y2])
    
    return anchors