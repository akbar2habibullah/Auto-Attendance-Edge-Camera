import cv2
import numpy as np
import logging
from core.utils import face_alignment

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
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.input_size = (112, 112)
        self.is_rknn = model_path.endswith('.rknn')

        if self.is_rknn and RKNN_AVAILABLE:
            self._init_rknn()
        elif ONNX_AVAILABLE:
            self._init_onnx()
        else:
            raise RuntimeError("No suitable runtime for ArcFace")

    def _init_rknn(self):
        logger.info(f"Loading ArcFace RKNN: {self.model_path}")
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.model_path) != 0:
            raise RuntimeError("RKNN Load Failed")
        if self.rknn.init_runtime() != 0:
            raise RuntimeError("RKNN Init Runtime Failed")

    def _init_onnx(self):
        logger.info(f"Loading ArcFace ONNX: {self.model_path}")
        self.session = InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, image: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """
        Extract feature embedding from image using landmarks (kps).
        Returns a 512-d normalized vector.
        """
        # 1. Align/Crop Face
        aligned_face = face_alignment(image, kps)

        # 2. Inference
        if self.is_rknn and RKNN_AVAILABLE:
            # RKNN expects RGB, Uint8. 
            rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            input_blob = np.expand_dims(rgb_face, axis=0)
            
            outputs = self.rknn.inference(inputs=[input_blob], data_format='nhwc')
            
            # SAFE OUTPUT EXTRACTION
            # outputs[0] is the tensor. It might be (1, 512) or (512,)
            embedding = outputs[0]
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
        else:
            # ONNX standard preprocessing
            blob = cv2.dnn.blobFromImage(
                aligned_face, 1.0/127.5, self.input_size, 
                (127.5, 127.5, 127.5), swapRB=True
            )
            outputs = self.session.run(None, {self.input_name: blob})
            embedding = outputs[0].flatten()

        # 3. Normalize (L2) - Crucial for Cosine Similarity
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm