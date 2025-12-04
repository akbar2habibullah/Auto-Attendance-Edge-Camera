import cv2
import numpy as np
from skimage.transform import SimilarityTransform

# Standard ArcFace reference points (112x112)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def estimate_norm(landmark: np.ndarray, image_size: int = 112):
    """
    Calculate the Affine Transform Matrix to align face.
    Matches original repo logic exactly.
    """
    assert landmark.shape == (5, 2)
    
    # Logic to handle 112x112 vs 128x128
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    # Adjust reference alignment based on ratio
    alignment = REFERENCE_LANDMARKS * ratio
    alignment[:, 0] += diff_x

    tform = SimilarityTransform()
    tform.estimate(landmark, alignment)
    M = tform.params[0:2, :]
    return M

def face_alignment(image: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """Crop and align face to 112x112."""
    M = estimate_norm(landmark, 112)
    warped = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
    return warped

def distance2bbox(points, distance, max_shape=None):
    """Decode SCRFD distance prediction to bounding box [x1, y1, x2, y2]."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode SCRFD distance prediction to keypoints (landmarks)."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def draw_bbox_info(frame, bbox, similarity, name, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox[:4]) 
    label = f"{name}: {similarity:.2f}"
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    rect_x_start = x2 + 5
    rect_x_end = rect_x_start + 10
    rect_y_end = y2
    rect_height = int(similarity * (y2 - y1))
    rect_y_start = rect_y_end - rect_height
    try:
        cv2.rectangle(frame, (rect_x_start, rect_y_start), (rect_x_end, rect_y_end), color, cv2.FILLED)
    except:
        pass

def draw_bbox_unknown(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(frame, "Unknown", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)