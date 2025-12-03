import numpy as np
import cv2
import logging
from typing import Tuple

from core.utils import distance2bbox, distance2kps

# Handle conditional import for PC development vs Edge deployment
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    # Only import ONNX if RKNN is missing (dev mode)
    try:
        import onnxruntime
        ONNX_AVAILABLE = True
    except ImportError:
        ONNX_AVAILABLE = False

logger = logging.getLogger("scrfd")

class SCRFD:
    def __init__(self, model_path: str, input_size: Tuple[int] = (640, 640), conf_thres=0.5, iou_thres=0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # SCRFD Specific Constants
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.center_cache = {}

        self.is_rknn = model_path.endswith('.rknn')
        
        if self.is_rknn and RKNN_AVAILABLE:
            self._init_rknn()
        elif ONNX_AVAILABLE:
            self._init_onnx()
        else:
            raise RuntimeError("No suitable runtime found (RKNNLite or ONNXRuntime)")

    def _init_rknn(self):
        logger.info(f"Loading RKNN Model: {self.model_path}")
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.model_path) != 0:
            raise RuntimeError("Failed to load RKNN model")
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError("Failed to init RKNN runtime")

    def _init_onnx(self):
        logger.info(f"Loading ONNX Model: {self.model_path}")
        self.session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image):
        """Resize and pad image to 640x640 while maintaining aspect ratio."""
        target_w, target_h = self.input_size
        h, w, _ = image.shape
        scale = min(target_w / w, target_h / h)
        
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh))
        
        # Create canvas
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:nh, :nw, :] = resized
        
        return padded, scale

    def forward(self, img_batch):
        if self.is_rknn and RKNN_AVAILABLE:
            # RKNN Input: [1, 640, 640, 3] RGB, Uint8
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_batch[0], cv2.COLOR_BGR2RGB)
            input_blob = np.expand_dims(img_rgb, axis=0)
            return self.rknn.inference(inputs=[input_blob], data_format='nhwc')
        else:
            # ONNX Input: [1, 3, 640, 640] BGR, Normalized, Float
            blob = cv2.dnn.blobFromImage(
                img_batch[0], 1.0/128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True
            )
            return self.session.run(None, {self.input_name: blob})

    def detect(self, image, max_num=0):
        # 1. Preprocess
        input_img, scale = self.preprocess(image)
        input_batch = np.expand_dims(input_img, axis=0)
        
        # 2. Inference
        outputs = self.forward(input_batch)
        
        # 3. Post-Process (Decode Strides)
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        input_h, input_w = self.input_size
        
        # The output order usually: [score8, score16, score32, bbox8, bbox16, bbox32, kps8, kps16, kps32]
        # But this depends heavily on export. Assuming standard SCRFD export order.
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.fmc] * stride
            kps_preds = outputs[idx + self.fmc * 2] * stride
            
            height = input_h // stride
            width = input_w // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # Generate anchors
                y, x = np.mgrid[:height, :width]
                anchor_centers = np.stack([x, y], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                self.center_cache[key] = anchor_centers

            # Filter by confidence
            pos_inds = np.where(scores >= self.conf_thres)[0]
            
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            kpss = distance2kps(anchor_centers, kps_preds)
            
            # Use Reshape to handle multiple anchors
            kpss = kpss.reshape((kpss.shape[0], -1, 2))

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list) / scale
        kpss = np.vstack(kpss_list) / scale
        
        # 4. NMS (Non-Maximum Suppression)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        keep = self.nms(pre_det, self.iou_thres)
        
        det = pre_det[keep, :]
        kpss = kpss[keep, :, :]

        # 5. Filter Max Num
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            bindex = np.argsort(area)[::-1][:max_num]
            det = det[bindex, :]
            kpss = kpss[bindex, :, :]
            
        return det, kpss

    def nms(self, dets, iou_thres):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= iou_thres)[0]
            order = order[indices + 1]

        return keep