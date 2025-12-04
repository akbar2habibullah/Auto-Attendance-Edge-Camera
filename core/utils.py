import cv2
import numpy as np
from skimage.transform import SimilarityTransform

# Reference alignment for facial landmarks (ArcFace)
reference_alignment: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ],
    dtype=np.float32
)

def estimate_norm(landmark: np.ndarray, image_size: int = 112):
    assert landmark.shape == (5, 2)
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    alignment = reference_alignment * ratio
    alignment[:, 0] += diff_x

    transform = SimilarityTransform()
    transform.estimate(landmark, alignment)
    return transform.params[0:2, :], np.linalg.inv(transform.params)[0:2, :]

def face_alignment(image: np.ndarray, landmark: np.ndarray, image_size: int = 112):
    M, M_inv = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
    return warped, M_inv

def distance2bbox(points, distance, max_shape=None):
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

# --- Visualization Helpers (Kept for your Video Feed) ---

def draw_bbox_info(frame, bbox, similarity, name, color=(0, 255, 0)):
    # bbox usually comes as [x1, y1, x2, y2, score]
    x1, y1, x2, y2 = map(int, bbox[:4])

    label = f"{name}: {similarity:.2f}"
    # Ensure text is readable
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Similarity Bar
    rect_height = int(similarity * (y2 - y1))
    cv2.rectangle(frame, (x2+5, y2-rect_height), (x2+10, y2), color, cv2.FILLED)

def draw_bbox_unknown(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(frame, "Unknown", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)