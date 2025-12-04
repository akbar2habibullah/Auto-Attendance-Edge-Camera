import cv2
import numpy as np
import logging
from core.utils import face_alignment

# Handle Imports
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False

try:
    from onnxruntime import InferenceSession
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger("arcface")

class ArcFace:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.input_size = (112, 112)
        self.normalization_mean = 127.5
        self.normalization_scale = 127.5
        
        self.is_rknn = model_path.endswith('.rknn')
        logger.info(f"Initializing ArcFace from {self.model_path}")

        if self.is_rknn:
            if not RKNN_AVAILABLE: raise ImportError("rknnlite not installed")
            self._init_rknn()
        else:
            if not ONNX_AVAILABLE: raise ImportError("onnxruntime not installed")
            self._init_onnx()

    def _init_rknn(self):
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.model_path) != 0: raise RuntimeError("RKNN Load Failed")
        if self.rknn.init_runtime() != 0: raise RuntimeError("RKNN Init Failed")

    def _init_onnx(self):
        self.session = InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess_onnx(self, face_image: np.ndarray) -> np.ndarray:
        resized_face = cv2.resize(face_image, self.input_size)
        face_blob = cv2.dnn.blobFromImage(
            resized_face, 1.0 / self.normalization_scale, self.input_size,
            (self.normalization_mean,)*3, swapRB=True
        )
        return face_blob

    def preprocess_rknn(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess for RKNN: Resize -> RGB -> Expand Dims
        """
        resized_face = cv2.resize(face_image, self.input_size)
        # Convert BGR (OpenCV) to RGB
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        # RKNN Lite with data_format='nhwc' expects [1, H, W, C]
        face_blob = np.expand_dims(rgb_face, axis=0)
        return face_blob

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        if image is None or landmarks is None:
            raise ValueError("Image/Landmarks None")

        try:
            # 1. Align the face
            aligned_face, _ = face_alignment(image, landmarks)

            # 2. Inference
            if self.is_rknn:
                face_blob = self.preprocess_rknn(aligned_face)
                outputs = self.rknn.inference(inputs=[face_blob], data_format='nhwc')
                embedding = outputs[0]
            else:
                face_blob = self.preprocess_onnx(aligned_face)
                outputs = self.session.run(self.output_names, {self.input_name: face_blob})
                embedding = outputs[0]

            # 3. Post-process (Flatten)
            embedding = embedding.flatten()

            # 4. Normalize (L2) - Crucial for Cosine Similarity
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return embedding
            normalized_embedding = embedding / norm
            return normalized_embedding

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise