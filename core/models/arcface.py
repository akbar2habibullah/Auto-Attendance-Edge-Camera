import cv2
import numpy as np
import logging
from ..utils import face_alignment

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
    def __init__(self, model_path):
        self.model_path = model_path
        self.input_size = (112, 112)
        self.is_rknn = model_path.endswith('.rknn')
        
        if self.is_rknn:
            self._init_rknn()
        else:
            self._init_onnx()

    def _init_rknn(self):
        logger.info(f"Loading ArcFace RKNN: {self.model_path}")
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.model_path) != 0: raise Exception("Load RKNN failed")
        if self.rknn.init_runtime() != 0: raise Exception("Init RKNN failed")

    def _init_onnx(self):
        logger.info(f"Loading ArcFace ONNX: {self.model_path}")
        self.session = InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, image, kps):
        # 1. Align
        aligned_face, _ = face_alignment(image, kps)
        
        # 2. Inference
        if self.is_rknn:
            # RKNN: Resize -> BGR2RGB -> Expand Dims -> Uint8
            # Note: No div/255 because it was baked into model
            input_blob = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            input_blob = np.expand_dims(input_blob, axis=0)
            outputs = self.rknn.inference(inputs=[input_blob], data_format='nhwc')
            embedding = outputs[0]
        else:
            # ONNX: Standard float32 NCHW
            blob = cv2.dnn.blobFromImage(aligned_face, 1.0/127.5, self.input_size, (127.5, 127.5, 127.5), swapRB=True)
            outputs = self.session.run(None, {self.input_name: blob})
            embedding = outputs[0]

        # 3. Post-Process (Normalize)
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm