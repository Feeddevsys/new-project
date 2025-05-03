#!/usr/bin/env python3
import os
import cv2
import numpy as np
import glob

# Directory where known faces are stored
known_faces_dir = "data/images/"
os.makedirs(known_faces_dir, exist_ok=True)

def get_image_files(directory):
    """Get all image files from directory"""
    types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
    files = []
    for ext in types:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return files

def load_known_faces(face_detector, face_recognizer):
    """Load known faces from directory and extract features"""
    known_faces = {}
    for file in get_image_files(known_faces_dir):
        image = cv2.imread(file)
        if image is None:
            continue
            
        # Detect and align faces
        height, width = image.shape[:2]
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(image)
        
        if faces is not None and len(faces) > 0:
            # Extract features from first detected face
            aligned_face = face_recognizer.alignCrop(image, faces[0])
            features = face_recognizer.feature(aligned_face)
            
            # Use filename (without extension) as ID
            face_id = os.path.splitext(os.path.basename(file))[0]
            known_faces[face_id] = features
            
    return known_faces

def recognize_face(frame, face_detector, face_recognizer, known_faces, threshold=0.5):
    """Detect and recognize faces in frame"""
    height, width = frame.shape[:2]
    face_detector.setInputSize((width, height))
    
    # Detect faces
    _, faces = face_detector.detect(frame)
    if faces is None:
        return None, None
        
    results = []
    for face in faces:
        # Align and extract features
        aligned_face = face_recognizer.alignCrop(frame, face)
        features = face_recognizer.feature(aligned_face)
        
        # Compare with known faces
        best_match = None
        best_score = 0
        
        for face_id, known_features in known_faces.items():
            score = face_recognizer.match(features, known_features, 
                                        cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_match = face_id
                
        if best_score >= threshold:
            results.append((best_match, best_score, face))
        else:
            results.append(("unknown", best_score, face))
            
    return results

def main():
    # Load models
    face_detector = cv2.FaceDetectorYN_create(
        "models/face_detection_yunet_2023mar_fp16.onnx", "", (0, 0))
    face_detector.setScoreThreshold(0.87)
    
    face_recognizer = cv2.FaceRecognizerSF_create(
        "models/face_recognizer_fast_fp16.onnx", "")
    
    # Load known faces
    known_faces = load_known_faces(face_detector, face_recognizer)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Recognize faces
        recognitions = recognize_face(frame, face_detector, face_recognizer, known_faces)
        
        # Draw results
        for face_id, score, face in (recognitions or []):
            box = list(map(int, face[:4]))
            confidence = face[-1]
            
            # Draw rectangle
            color = (0, 255, 0) if face_id != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (box[0], box[1]), 
                         (box[0]+box[2], box[1]+box[3]), color, 2)
            
            # Draw label
            label = f"{face_id} ({score:.2f})" if face_id != "unknown" else "unknown"
            cv2.putText(frame, label, (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display
        cv2.imshow("Face Recognition", frame)
        
        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()