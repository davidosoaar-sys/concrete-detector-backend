"""
FastAPI Backend for 3D Concrete Print Monitoring System
Handles YOLO object detection using ONNX Runtime
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Dict, Any
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Concrete Print Detection API",
    description="Real-time defect detection for 3D concrete printing",
    version="1.0.0"
)

# CORS configuration - adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
ort_session = None
class_names = ["bead_break", "extrusion_issue", "good_bead", "layer_failure"]
input_size = (640, 640)  # YOLOv8 standard input size


def load_model():
    """Load ONNX model on startup"""
    global ort_session
    try:
        model_path = "best.onnx"
        
        # Configure ONNX Runtime for CPU
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Input name: {ort_session.get_inputs()[0].name}")
        logger.info(f"Input shape: {ort_session.get_inputs()[0].shape}")
        logger.info(f"Output names: {[output.name for output in ort_session.get_outputs()]}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for YOLO model
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed numpy array in format (1, 3, 640, 640)
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    img_resized = image.resize(input_size, Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img_resized).astype(np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Transpose to CHW format (channels first)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def postprocess_detections(
    output: np.ndarray,
    original_size: tuple,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Process YOLO output to extract detections
    
    Args:
        output: Raw model output
        original_size: Original image size (width, height)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    # YOLOv8 output format: (1, 84, 8400) or (1, num_classes + 4, num_predictions)
    # Shape: [batch, 4 + num_classes, num_boxes]
    output = output[0]  # Remove batch dimension
    
    # Transpose to [num_boxes, 4 + num_classes]
    if output.shape[0] < output.shape[1]:
        output = output.transpose()
    
    # Extract boxes and class scores
    boxes = output[:, :4]  # x_center, y_center, width, height
    class_scores = output[:, 4:]  # class probabilities
    
    # Get max class score and class id for each box
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # Filter by confidence threshold
    mask = max_scores > conf_threshold
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return detections
    
    # Convert boxes from center format to corner format
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # Scale boxes to original image size
    scale_x = original_size[0] / input_size[0]
    scale_y = original_size[1] / input_size[1]
    
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        max_scores.tolist(),
        conf_threshold,
        iou_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        
        for idx in indices:
            detection = {
                "class_id": int(class_ids[idx]),
                "class_name": class_names[int(class_ids[idx])],
                "confidence": float(max_scores[idx]),
                "bbox": {
                    "x1": float(boxes_xyxy[idx][0]),
                    "y1": float(boxes_xyxy[idx][1]),
                    "x2": float(boxes_xyxy[idx][2]),
                    "y2": float(boxes_xyxy[idx][3])
                }
            }
            detections.append(detection)
    
    return detections


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "3D Concrete Print Detection API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": ort_session is not None,
        "classes": class_names
    }


@app.post("/detect")
async def detect_defects(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Detect defects in uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
        confidence: Confidence threshold (0.0-1.0)
        iou_threshold: IoU threshold for NMS
        
    Returns:
        JSON with detections
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        original_size = image.size  # (width, height)
        
        logger.info(f"Processing image: {file.filename}, size: {original_size}")
        
        # Preprocess
        input_tensor = preprocess_image(image)
        
        # Run inference
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})
        
        # Postprocess
        detections = postprocess_detections(
            outputs[0],
            original_size,
            conf_threshold=confidence,
            iou_threshold=iou_threshold
        )
        
        # Calculate summary statistics
        summary = {
            "total_detections": len(detections),
            "by_class": {}
        }
        
        for class_name in class_names:
            count = sum(1 for d in detections if d["class_name"] == class_name)
            summary["by_class"][class_name] = count
        
        logger.info(f"Detected {len(detections)} objects")
        
        return JSONResponse(content={
            "success": True,
            "image_size": {"width": original_size[0], "height": original_size[1]},
            "detections": detections,
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/batch-detect")
async def batch_detect(
    files: List[UploadFile] = File(...),
    confidence: float = 0.25
):
    """
    Process multiple images in batch
    
    Args:
        files: List of image files
        confidence: Confidence threshold
        
    Returns:
        JSON with results for each image
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            original_size = image.size
            
            input_tensor = preprocess_image(image)
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: input_tensor})
            
            detections = postprocess_detections(
                outputs[0],
                original_size,
                conf_threshold=confidence
            )
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": detections,
                "count": len(detections)
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)