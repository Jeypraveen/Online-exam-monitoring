import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import os
from PIL import Image, ImageTk
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
from ultralytics import YOLO
from collections import deque, Counter, defaultdict
import warnings
import csv
warnings.filterwarnings('ignore')

class ExamProctoringSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Exam Proctoring System")
        self.root.geometry("1200x800")
        
        # Initialize models
        self.setup_models()
        
        # Initialize variables
        self.is_monitoring = False
        self.is_screening = False
        self.cap = None
        self.known_embeddings = None
        self.name_mapping = None
        self.student_name = "Unknown"
        self.student_id = None
        self.violation_log = []
        self.cognitive_states = {'attentive': 0, 'distracted': 0, 'drowsy': 0, 'absent': 0}
        self.total_frames = 0
        self.session_start_time = None
        
        # For Temporal Smoothing
        self.HISTORY_LENGTH = 10
        self.cognitive_state_history = deque(maxlen=self.HISTORY_LENGTH)
        
        # Setup GUI
        self.setup_gui()
        
        # Load face embeddings
        self.load_face_data()
    
    def setup_models(self):
        """Initialize all AI models"""
        try:
            # Face recognition models
            self.mtcnn = MTCNN(keep_all=True, device='cpu')
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
            
            # MediaPipe for face landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # YOLOv8 for object detection
            self.yolo_model = YOLO('yolov8n.pt')
            
            # Banned objects class IDs in COCO dataset
            self.banned_objects = {
                67: 'cell phone', 73: 'laptop', 74: 'mouse', 75: 'remote',
                76: 'keyboard', 84: 'book'
            }
            
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load models: {str(e)}")
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="AI-Powered Exam Proctoring System",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Live Monitoring", padding="5")
        self.video_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="5")
        status_frame.grid(row=1, column=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N))
        
        self.student_label = ttk.Label(status_frame, text="Student: Not Identified")
        self.student_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.cognitive_label = ttk.Label(status_frame, text="State: Monitoring Off")
        self.cognitive_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        self.violations_label = ttk.Label(status_frame, text="Violations: 0")
        self.violations_label.grid(row=3, column=0, sticky=tk.W, pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Monitoring",
                                   command=self.start_screening, state='normal')
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring",
                                  command=self.stop_monitoring, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        self.report_btn = ttk.Button(control_frame, text="Generate Report",
                                    command=self.generate_report)
        self.report_btn.grid(row=0, column=2, padx=5)
        
        # Log display
        log_frame = ttk.LabelFrame(main_frame, text="Violation Log", padding="5")
        log_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = tk.Text(log_frame, height=10, width=80)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def load_face_data(self):
        """Load face embeddings and name mapping using FaceNet format"""
        try:
            # First try to load from standard locations
            embeddings_path = "embeddings/face_embeddings.npy"
            names_path = "embeddings/name_mapping.npy"
            
            # If not found, try alternative paths
            if not os.path.exists(embeddings_path) or not os.path.exists(names_path):
                # Try in current directory
                embeddings_path = "face_embeddings.npy"
                names_path = "name_mapping.npy"
                
                if not os.path.exists(embeddings_path) or not os.path.exists(names_path):
                    # Try absolute paths
                    embeddings_path = r'C:\Projects\vvvvvv\face_embeddings.npy'
                    names_path = r'C:\Projects\vvvvvv\name_mapping.npy'
            
            # Load the embeddings
            if os.path.exists(embeddings_path) and os.path.exists(names_path):
                self.known_embeddings = np.load(embeddings_path, allow_pickle=True).item()
                self.name_mapping = np.load(names_path, allow_pickle=True).item()
                self.log_message(f"Face recognition data loaded successfully. Registered students: {len(self.known_embeddings)}")
            else:
                self.log_message("Warning: Face recognition files not found in any expected location")
                
        except Exception as e:
            self.log_message(f"Error loading face data: {str(e)}")
            self.known_embeddings = None
            self.name_mapping = None
    
    def log_message(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Add to violation log if monitoring
        if self.is_monitoring and "Warning" in message:
            self.violation_log.append({
                'timestamp': datetime.now(),
                'violation': message,
                'student': self.student_name
            })
    
    def start_screening(self):
        """Start the pre-exam screening process"""
        self.log_message("Starting pre-exam screening...")
        
        # Create a new window for pre-exam screening
        self.screening_window = tk.Toplevel(self.root)
        self.screening_window.title("Pre-Exam Screening")
        self.screening_window.geometry("800x600")
        
        # Add instructions
        instructions = ttk.Label(self.screening_window, 
                               text="Please look at the camera for identification.\n"
                                    "The system will verify your identity before allowing you to take the exam.",
                               font=('Arial', 12))
        instructions.pack(pady=20)
        
        # Video frame for screening
        self.screening_frame = ttk.Label(self.screening_window)
        self.screening_frame.pack(pady=10)
        
        # Status label
        self.screening_status = ttk.Label(self.screening_window, 
                                        text="Status: Waiting for face detection...",
                                        font=('Arial', 12))
        self.screening_status.pack(pady=10)
        
        # Start screening
        self.screening_cap = cv2.VideoCapture(0)
        if not self.screening_cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot access camera for screening")
            self.screening_window.destroy()
            return
            
        self.is_screening = True
        self.screening_countdown = 15  # Seconds to wait for identification
        self.screening_status.config(text=f"Status: Looking for registered student ({self.screening_countdown}s remaining)")
        
        # Start the screening loop
        self.screening_window.protocol("WM_DELETE_WINDOW", self.cancel_screening)
        self.screening_loop()
    
    def cancel_screening(self):
        """Cancel the pre-exam screening"""
        self.is_screening = False
        if self.screening_cap:
            self.screening_cap.release()
        self.screening_window.destroy()
        self.log_message("Pre-exam screening cancelled")
    
    def screening_loop(self):
        """Loop for pre-exam screening"""
        if not self.is_screening:
            return
            
        try:
            ret, frame = self.screening_cap.read()
            if not ret:
                self.screening_status.config(text="Error: Failed to capture frame")
                self.screening_window.after(100, self.screening_loop)
                return
                
            # Process frame for face recognition
            self.screening_countdown -= 0.03  # Adjust based on loop timing
            
            # Display frame
            display_frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            self.screening_frame.configure(image=photo)
            self.screening_frame.image = photo  # Keep reference
            
            # Check for faces and identify student
            student_id, confidence = self.recognize_face_facenet(frame)
            
            if student_id is not None and confidence > 85:
                # Student identified successfully
                name = self.name_mapping.get(student_id, "Unknown")
                self.student_name = name
                self.student_id = student_id
                self.is_screening = False
                
                # Release camera and close screening window
                self.screening_cap.release()
                self.screening_window.destroy()
                
                # Start actual monitoring
                self.start_monitoring()
                
                self.log_message(f"Student verified: {name} (ID: {student_id}, Confidence: {confidence:.1f}%)")
                self.log_message("Proceeding with exam monitoring...")
                return
            
            # Update status
            if self.screening_countdown > 0:
                self.screening_status.config(text=f"Status: Looking for registered student ({int(self.screening_countdown)}s remaining)")
            else:
                # Time's up - no student identified
                self.is_screening = False
                self.screening_cap.release()
                self.screening_window.destroy()
                messagebox.showerror("Screening Failed", 
                                    "Could not verify your identity. Only registered students can take the exam.")
                self.log_message("Pre-exam screening failed: Student not identified")
                
            # Continue screening loop
            self.screening_window.after(30, self.screening_loop)
            
        except Exception as e:
            self.log_message(f"Screening error: {str(e)}")
            self.cancel_screening()
    
    def recognize_face_facenet(self, frame):
        """Recognize face using FaceNet embeddings with MTCNN detection and alignment"""
        try:
            # Convert frame to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Detect faces using MTCNN
            boxes, probs = self.mtcnn.detect(pil_image)
            
            if boxes is not None and len(boxes) > 0:
                # Select the face with the highest detection probability
                best_face_idx = np.argmax(probs)
                best_box = boxes[best_face_idx]
                best_prob = probs[best_face_idx]
                
                if best_prob > 0.9:  # Threshold for detection confidence
                    # Extract aligned face tensor using MTCNN
                    face_tensor = self.mtcnn.extract(pil_image, [best_box], save_path=None)
                    
                    if face_tensor is not None:
                        # Prepare tensor for FaceNet (add batch dimension)
                        face_tensor = face_tensor[0].unsqueeze(0)
                        
                        # Compute embedding with FaceNet
                        with torch.no_grad():
                            embedding = self.resnet(face_tensor).numpy().flatten()
                            # Normalize the embedding
                            embedding = embedding / np.linalg.norm(embedding)
                        
                        # Compare with stored embeddings
                        best_match_id = None
                        best_similarity = -1
                        
                        for student_id, stored_embedding in self.known_embeddings.items():
                            # Calculate cosine similarity (dot product since both are normalized)
                            similarity = np.dot(embedding, stored_embedding)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match_id = student_id
                        
                        # Convert similarity to confidence (0-100 scale)
                        confidence = best_similarity * 100
                        return best_match_id, confidence
            
            # No face detected or insufficient confidence
            return None, 0
        except Exception as e:
            self.log_message(f"Recognition error: {str(e)}")
            return None, 0
    
    def start_monitoring(self):
        """Start the monitoring process"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Cannot access camera")
                return
            
            self.is_monitoring = True
            self.session_start_time = datetime.now()
            self.violation_log = []
            self.cognitive_states = {'attentive': 0, 'distracted': 0, 'drowsy': 0, 'absent': 0}
            self.total_frames = 0
            self.cognitive_state_history.clear()  # Reset history buffer
            
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_label.config(text="Status: Monitoring Active")
            
            # Update student label with verified name
            self.student_label.config(text=f"Student: {self.student_name}")
            
            self.log_message("Monitoring session started")
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: Monitoring Stopped")
        
        self.log_message("Monitoring session ended")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.total_frames += 1
                
                # Process frame
                processed_frame = self.process_frame(frame.copy())
                
                # Update GUI
                self.update_video_display(processed_frame)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.03)
                
            except Exception as e:
                self.log_message(f"Error in monitoring loop: {str(e)}")
                break
    
    def process_frame(self, frame):
        """Process each frame for violations and cognitive states"""
        height, width = frame.shape[:2]
        
        # First get face ROI and landmarks
        faces = self.detect_face_roi(frame)
        
        # Analyze face recognition and cognitive states using the ROI
        self.analyze_face_and_cognitive_state(frame, faces)
        
        # Object detection
        self.detect_banned_objects(frame)
        
        # Person counting
        self.count_persons(frame)
        
        return frame
    
    def detect_face_roi(self, frame):
        """Detect face and return ROI with landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        faces = []
        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # Get bounding box from landmarks
                h, w = frame.shape[:2]
                x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
                
                x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
                
                # Create face ROI
                face_roi = frame[y_min:y_max, x_min:x_max]
                
                # Store face data
                faces.append({
                    'roi': face_roi,
                    'landmarks': face_landmarks,
                    'bbox': (x_min, y_min, x_max, y_max)
                })
                
                # Draw face bounding box for visualization
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return faces
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for drowsiness detection with 6 landmark points"""
        try:
            # Ensure eye_landmarks has the expected 6 points
            if len(eye_landmarks) != 6:
                return 0.3  # Default value
            
            # Calculate distances
            vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            
            if horizontal == 0:
                return 0.3  # Default value
            
            return (vertical_1 + vertical_2) / (2.0 * horizontal)
        except Exception as e:
            print(f"EAR calculation error: {e}")
            return 0.3  # Default value if calculation fails
    
    def get_cognitive_state(self, landmarks, w, h):
        """Determine cognitive state based on landmarks with improved accuracy"""
        try:
            # Convert normalized MediaPipe landmarks to pixel coordinates
            pixel_landmarks = []
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pixel_landmarks.append((x, y))
            
            # Define landmark indices for cognitive state detection
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            nose_tip_idx = 1
            face_width_lm_left = 33
            face_width_lm_right = 263
            
            # Check if we have enough landmarks
            max_required_idx = max(left_eye_indices + right_eye_indices + 
                                 [nose_tip_idx, face_width_lm_left, face_width_lm_right])
            if len(pixel_landmarks) <= max_required_idx:
                return 'absent'
            
            # Get eye landmarks
            left_eye = [pixel_landmarks[i] for i in left_eye_indices]
            right_eye = [pixel_landmarks[i] for i in right_eye_indices]
            
            # Calculate eye aspect ratio
            avg_ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0
            
            # Drowsiness detection
            if avg_ear < 0.2:
                return 'drowsy'
            
            # Head pose estimation
            nose_tip = pixel_landmarks[nose_tip_idx]
            face_width = abs(pixel_landmarks[face_width_lm_left][0] - 
                            pixel_landmarks[face_width_lm_right][0])
            
            if face_width == 0:
                return 'attentive'
                
            eye_center_x = (pixel_landmarks[face_width_lm_left][0] + 
                           pixel_landmarks[face_width_lm_right][0]) / 2.0
            gaze_offset = (nose_tip[0] - eye_center_x) / face_width
            
            # Distraction detection
            return 'attentive' if abs(gaze_offset) < 0.1 else 'distracted'
            
        except IndexError:
            return 'absent'
        except Exception as e:
            print(f"Cognitive state error: {e}")
            return 'attentive'
    
    def analyze_face_and_cognitive_state(self, frame, faces):
        """Analyze face recognition and cognitive states with improved cognitive detection"""
        if not faces:
            # No face detected
            self.cognitive_states['absent'] += 1
            self.total_frames += 1
            
            # Only update UI periodically to avoid flickering
            if self.total_frames % 15 == 0:
                self.cognitive_label.config(text="State: No Face Detected")
                if self.total_frames % 45 == 0:  # Log less frequently
                    self.log_message("Warning: No face detected - student may have left")
            return
        
        if len(faces) > 1:
            self.log_message("Warning: Multiple faces detected - possible cheating")
        
        # Process the primary face (first detected)
        face_data = faces[0]
        landmarks = face_data['landmarks'].landmark
        h, w = frame.shape[:2]
        
        # Determine cognitive state
        current_state = self.get_cognitive_state(landmarks, w, h)
        
        # Add current state to history for smoothing (prevents flickering)
        self.cognitive_state_history.append(current_state)
        
        # Determine dominant state from history
        most_common_state = Counter(self.cognitive_state_history).most_common(1)[0][0]
        
        # Update cognitive state counters
        self.cognitive_states[most_common_state] += 1
        self.total_frames += 1
        
        # Update UI with smoothed state
        if most_common_state == 'drowsy':
            self.cognitive_label.config(text="State: Drowsy (Eyes Closed)")
            if self.total_frames % 15 == 0:  # Log every 15 frames
                self.log_message("Warning: Student appears drowsy (eyes closed)")
        elif most_common_state == 'distracted':
            self.cognitive_label.config(text="State: Distracted (Looking Away)")
            if self.total_frames % 20 == 0:  # Log every 20 frames
                self.log_message("Warning: Student looking away from screen")
        else:
            self.cognitive_label.config(text="State: Attentive")
        
        # Visual feedback on frame
        self.visualize_cognitive_state(frame, face_data['bbox'], most_common_state)
    
    def visualize_cognitive_state(self, frame, bbox, state):
        """Visualize cognitive state on the video frame"""
        x_min, y_min, x_max, y_max = bbox
        
        # Draw state-specific visualization
        if state == 'drowsy':
            color = (0, 0, 255)  # Red
            cv2.putText(frame, "DROWSY (EYES CLOSED)", (x_min, y_min - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif state == 'distracted':
            color = (0, 165, 255)  # Orange
            cv2.putText(frame, "DISTRACTED", (x_min, y_min - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:  # attentive
            color = (0, 255, 0)  # Green
        
        # Draw face bounding box with state color
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    def detect_banned_objects(self, frame):
        """Detect banned objects using YOLOv8"""
        try:
            results = self.yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id in self.banned_objects and confidence > 0.5:
                            object_name = self.banned_objects[class_id]
                            self.log_message(f"Warning: Banned object detected - {object_name}")
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{object_name}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            pass  # Silently handle object detection errors
    
    import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import os
from PIL import Image, ImageTk
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
from ultralytics import YOLO
from collections import deque, Counter, defaultdict
import warnings
import csv
warnings.filterwarnings('ignore')

class ExamProctoringSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Exam Proctoring System")
        self.root.geometry("1200x800")
        
        # Initialize models
        self.setup_models()
        
        # Initialize variables
        self.is_monitoring = False
        self.is_screening = False
        self.cap = None
        self.known_embeddings = None
        self.name_mapping = None
        self.student_name = "Unknown"
        self.student_id = None
        self.violation_log = []
        self.cognitive_states = {'attentive': 0, 'distracted': 0, 'drowsy': 0, 'absent': 0}
        self.total_frames = 0
        self.session_start_time = None
        
        # For Temporal Smoothing
        self.HISTORY_LENGTH = 10
        self.cognitive_state_history = deque(maxlen=self.HISTORY_LENGTH)
        
        # Setup GUI
        self.setup_gui()
        
        # Load face embeddings
        self.load_face_data()
    
    def setup_models(self):
        """Initialize all AI models"""
        try:
            # Face recognition models
            self.mtcnn = MTCNN(keep_all=True, device='cpu')
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
            
            # MediaPipe for face landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # YOLOv8 for object detection
            self.yolo_model = YOLO('yolov8n.pt')
            
            # Banned objects class IDs in COCO dataset
            self.banned_objects = {
                67: 'cell phone', 73: 'laptop', 74: 'mouse', 75: 'remote',
                76: 'keyboard', 84: 'book'
            }
            
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load models: {str(e)}")
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="AI-Powered Exam Proctoring System",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Live Monitoring", padding="5")
        self.video_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="5")
        status_frame.grid(row=1, column=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N))
        
        self.student_label = ttk.Label(status_frame, text="Student: Not Identified")
        self.student_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.cognitive_label = ttk.Label(status_frame, text="State: Monitoring Off")
        self.cognitive_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        self.violations_label = ttk.Label(status_frame, text="Violations: 0")
        self.violations_label.grid(row=3, column=0, sticky=tk.W, pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Monitoring",
                                   command=self.start_screening, state='normal')
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring",
                                  command=self.stop_monitoring, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        self.report_btn = ttk.Button(control_frame, text="Generate Report",
                                    command=self.generate_report)
        self.report_btn.grid(row=0, column=2, padx=5)
        
        # Log display
        log_frame = ttk.LabelFrame(main_frame, text="Violation Log", padding="5")
        log_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = tk.Text(log_frame, height=10, width=80)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def load_face_data(self):
        """Load face embeddings and name mapping using FaceNet format"""
        try:
            # First try to load from standard locations
            embeddings_path = "embeddings/face_embeddings.npy"
            names_path = "embeddings/name_mapping.npy"
            
            # If not found, try alternative paths
            if not os.path.exists(embeddings_path) or not os.path.exists(names_path):
                # Try in current directory
                embeddings_path = "face_embeddings.npy"
                names_path = "name_mapping.npy"
                
                if not os.path.exists(embeddings_path) or not os.path.exists(names_path):
                    # Try absolute paths
                    embeddings_path = r'C:\Projects\vvvvvv\face_embeddings.npy'
                    names_path = r'C:\Projects\vvvvvv\name_mapping.npy'
            
            # Load the embeddings
            if os.path.exists(embeddings_path) and os.path.exists(names_path):
                self.known_embeddings = np.load(embeddings_path, allow_pickle=True).item()
                self.name_mapping = np.load(names_path, allow_pickle=True).item()
                self.log_message(f"Face recognition data loaded successfully. Registered students: {len(self.known_embeddings)}")
            else:
                self.log_message("Warning: Face recognition files not found in any expected location")
                
        except Exception as e:
            self.log_message(f"Error loading face data: {str(e)}")
            self.known_embeddings = None
            self.name_mapping = None
    
    def log_message(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Add to violation log if monitoring
        if self.is_monitoring and "Warning" in message:
            self.violation_log.append({
                'timestamp': datetime.now(),
                'violation': message,
                'student': self.student_name
            })
    
    def start_screening(self):
        """Start the pre-exam screening process"""
        self.log_message("Starting pre-exam screening...")
        
        # Create a new window for pre-exam screening
        self.screening_window = tk.Toplevel(self.root)
        self.screening_window.title("Pre-Exam Screening")
        self.screening_window.geometry("800x600")
        
        # Add instructions
        instructions = ttk.Label(self.screening_window, 
                               text="Please look at the camera for identification.\n"
                                    "The system will verify your identity before allowing you to take the exam.",
                               font=('Arial', 12))
        instructions.pack(pady=20)
        
        # Video frame for screening
        self.screening_frame = ttk.Label(self.screening_window)
        self.screening_frame.pack(pady=10)
        
        # Status label
        self.screening_status = ttk.Label(self.screening_window, 
                                        text="Status: Waiting for face detection...",
                                        font=('Arial', 12))
        self.screening_status.pack(pady=10)
        
        # Start screening
        self.screening_cap = cv2.VideoCapture(0)
        if not self.screening_cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot access camera for screening")
            self.screening_window.destroy()
            return
            
        self.is_screening = True
        self.screening_countdown = 15  # Seconds to wait for identification
        self.screening_status.config(text=f"Status: Looking for registered student ({self.screening_countdown}s remaining)")
        
        # Start the screening loop
        self.screening_window.protocol("WM_DELETE_WINDOW", self.cancel_screening)
        self.screening_loop()
    
    def cancel_screening(self):
        """Cancel the pre-exam screening"""
        self.is_screening = False
        if self.screening_cap:
            self.screening_cap.release()
        self.screening_window.destroy()
        self.log_message("Pre-exam screening cancelled")
    
    def screening_loop(self):
        """Loop for pre-exam screening"""
        if not self.is_screening:
            return
            
        try:
            ret, frame = self.screening_cap.read()
            if not ret:
                self.screening_status.config(text="Error: Failed to capture frame")
                self.screening_window.after(100, self.screening_loop)
                return
                
            # Process frame for face recognition
            self.screening_countdown -= 0.03  # Adjust based on loop timing
            
            # Display frame
            display_frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            self.screening_frame.configure(image=photo)
            self.screening_frame.image = photo  # Keep reference
            
            # Check for faces and identify student
            student_id, confidence = self.recognize_face_facenet(frame)
            
            if student_id is not None and confidence > 85:
                # Student identified successfully
                name = self.name_mapping.get(student_id, "Unknown")
                self.student_name = name
                self.student_id = student_id
                self.is_screening = False
                
                # Release camera and close screening window
                self.screening_cap.release()
                self.screening_window.destroy()
                
                # Start actual monitoring
                self.start_monitoring()
                
                self.log_message(f"Student verified: {name} (ID: {student_id}, Confidence: {confidence:.1f}%)")
                self.log_message("Proceeding with exam monitoring...")
                return
            
            # Update status
            if self.screening_countdown > 0:
                self.screening_status.config(text=f"Status: Looking for registered student ({int(self.screening_countdown)}s remaining)")
            else:
                # Time's up - no student identified
                self.is_screening = False
                self.screening_cap.release()
                self.screening_window.destroy()
                messagebox.showerror("Screening Failed", 
                                    "Could not verify your identity. Only registered students can take the exam.")
                self.log_message("Pre-exam screening failed: Student not identified")
                
            # Continue screening loop
            self.screening_window.after(30, self.screening_loop)
            
        except Exception as e:
            self.log_message(f"Screening error: {str(e)}")
            self.cancel_screening()
    
    def recognize_face_facenet(self, frame):
        """Recognize face using FaceNet embeddings with MTCNN detection and alignment"""
        try:
            # Convert frame to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Detect faces using MTCNN
            boxes, probs = self.mtcnn.detect(pil_image)
            
            if boxes is not None and len(boxes) > 0:
                # Select the face with the highest detection probability
                best_face_idx = np.argmax(probs)
                best_box = boxes[best_face_idx]
                best_prob = probs[best_face_idx]
                
                if best_prob > 0.9:  # Threshold for detection confidence
                    # Extract aligned face tensor using MTCNN
                    face_tensor = self.mtcnn.extract(pil_image, [best_box], save_path=None)
                    
                    if face_tensor is not None:
                        # Prepare tensor for FaceNet (add batch dimension)
                        face_tensor = face_tensor[0].unsqueeze(0)
                        
                        # Compute embedding with FaceNet
                        with torch.no_grad():
                            embedding = self.resnet(face_tensor).numpy().flatten()
                            # Normalize the embedding
                            embedding = embedding / np.linalg.norm(embedding)
                        
                        # Compare with stored embeddings
                        best_match_id = None
                        best_similarity = -1
                        
                        for student_id, stored_embedding in self.known_embeddings.items():
                            # Calculate cosine similarity (dot product since both are normalized)
                            similarity = np.dot(embedding, stored_embedding)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match_id = student_id
                        
                        # Convert similarity to confidence (0-100 scale)
                        confidence = best_similarity * 100
                        return best_match_id, confidence
            
            # No face detected or insufficient confidence
            return None, 0
        except Exception as e:
            self.log_message(f"Recognition error: {str(e)}")
            return None, 0
    
    def start_monitoring(self):
        """Start the monitoring process"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Cannot access camera")
                return
            
            self.is_monitoring = True
            self.session_start_time = datetime.now()
            self.violation_log = []
            self.cognitive_states = {'attentive': 0, 'distracted': 0, 'drowsy': 0, 'absent': 0}
            self.total_frames = 0
            self.cognitive_state_history.clear()  # Reset history buffer
            
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status_label.config(text="Status: Monitoring Active")
            
            # Update student label with verified name
            self.student_label.config(text=f"Student: {self.student_name}")
            
            self.log_message("Monitoring session started")
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: Monitoring Stopped")
        
        self.log_message("Monitoring session ended")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.total_frames += 1
                
                # Process frame
                processed_frame = self.process_frame(frame.copy())
                
                # Update GUI
                self.update_video_display(processed_frame)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.03)
                
            except Exception as e:
                self.log_message(f"Error in monitoring loop: {str(e)}")
                break
    
    def process_frame(self, frame):
        """Process each frame for violations and cognitive states"""
        height, width = frame.shape[:2]
        
        # First get face ROI and landmarks
        faces = self.detect_face_roi(frame)
        
        # Analyze face recognition and cognitive states using the ROI
        self.analyze_face_and_cognitive_state(frame, faces)
        
        # Object detection
        self.detect_banned_objects(frame)
        
        # Person counting
        self.count_persons(frame)
        
        return frame
    
    def detect_face_roi(self, frame):
        """Detect face and return ROI with landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        faces = []
        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                # Get bounding box from landmarks
                h, w = frame.shape[:2]
                x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
                
                x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
                
                # Create face ROI
                face_roi = frame[y_min:y_max, x_min:x_max]
                
                # Store face data
                faces.append({
                    'roi': face_roi,
                    'landmarks': face_landmarks,
                    'bbox': (x_min, y_min, x_max, y_max)
                })
                
                # Draw face bounding box for visualization
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return faces
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for drowsiness detection with 6 landmark points"""
        try:
            # Ensure eye_landmarks has the expected 6 points
            if len(eye_landmarks) != 6:
                return 0.3  # Default value
            
            # Calculate distances
            vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            
            if horizontal == 0:
                return 0.3  # Default value
            
            return (vertical_1 + vertical_2) / (2.0 * horizontal)
        except Exception as e:
            print(f"EAR calculation error: {e}")
            return 0.3  # Default value if calculation fails
    
    def get_cognitive_state(self, landmarks, w, h):
        """Determine cognitive state based on landmarks with improved accuracy"""
        try:
            # Convert normalized MediaPipe landmarks to pixel coordinates
            pixel_landmarks = []
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pixel_landmarks.append((x, y))
            
            # Define landmark indices for cognitive state detection
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            nose_tip_idx = 1
            face_width_lm_left = 33
            face_width_lm_right = 263
            
            # Check if we have enough landmarks
            max_required_idx = max(left_eye_indices + right_eye_indices + 
                                 [nose_tip_idx, face_width_lm_left, face_width_lm_right])
            if len(pixel_landmarks) <= max_required_idx:
                return 'absent'
            
            # Get eye landmarks
            left_eye = [pixel_landmarks[i] for i in left_eye_indices]
            right_eye = [pixel_landmarks[i] for i in right_eye_indices]
            
            # Calculate eye aspect ratio
            avg_ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0
            
            # Drowsiness detection
            if avg_ear < 0.2:
                return 'drowsy'
            
            # Head pose estimation
            nose_tip = pixel_landmarks[nose_tip_idx]
            face_width = abs(pixel_landmarks[face_width_lm_left][0] - 
                            pixel_landmarks[face_width_lm_right][0])
            
            if face_width == 0:
                return 'attentive'
                
            eye_center_x = (pixel_landmarks[face_width_lm_left][0] + 
                           pixel_landmarks[face_width_lm_right][0]) / 2.0
            gaze_offset = (nose_tip[0] - eye_center_x) / face_width
            
            # Distraction detection
            return 'attentive' if abs(gaze_offset) < 0.1 else 'distracted'
            
        except IndexError:
            return 'absent'
        except Exception as e:
            print(f"Cognitive state error: {e}")
            return 'attentive'
    
    def analyze_face_and_cognitive_state(self, frame, faces):
        """Analyze face recognition and cognitive states with improved cognitive detection"""
        if not faces:
            # No face detected
            self.cognitive_states['absent'] += 1
            self.total_frames += 1
            
            # Only update UI periodically to avoid flickering
            if self.total_frames % 15 == 0:
                self.cognitive_label.config(text="State: No Face Detected")
                if self.total_frames % 45 == 0:  # Log less frequently
                    self.log_message("Warning: No face detected - student may have left")
            return
        
        if len(faces) > 1:
            self.log_message("Warning: Multiple faces detected - possible cheating")
        
        # Process the primary face (first detected)
        face_data = faces[0]
        landmarks = face_data['landmarks'].landmark
        h, w = frame.shape[:2]
        
        # Determine cognitive state
        current_state = self.get_cognitive_state(landmarks, w, h)
        
        # Add current state to history for smoothing (prevents flickering)
        self.cognitive_state_history.append(current_state)
        
        # Determine dominant state from history
        most_common_state = Counter(self.cognitive_state_history).most_common(1)[0][0]
        
        # Update cognitive state counters
        self.cognitive_states[most_common_state] += 1
        self.total_frames += 1
        
        # Update UI with smoothed state
        if most_common_state == 'drowsy':
            self.cognitive_label.config(text="State: Drowsy (Eyes Closed)")
            if self.total_frames % 15 == 0:  # Log every 15 frames
                self.log_message("Warning: Student appears drowsy (eyes closed)")
        elif most_common_state == 'distracted':
            self.cognitive_label.config(text="State: Distracted (Looking Away)")
            if self.total_frames % 20 == 0:  # Log every 20 frames
                self.log_message("Warning: Student looking away from screen")
        else:
            self.cognitive_label.config(text="State: Attentive")
        
        # Visual feedback on frame
        self.visualize_cognitive_state(frame, face_data['bbox'], most_common_state)
    
    def visualize_cognitive_state(self, frame, bbox, state):
        """Visualize cognitive state on the video frame"""
        x_min, y_min, x_max, y_max = bbox
        
        # Draw state-specific visualization
        if state == 'drowsy':
            color = (0, 0, 255)  # Red
            cv2.putText(frame, "DROWSY (EYES CLOSED)", (x_min, y_min - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif state == 'distracted':
            color = (0, 165, 255)  # Orange
            cv2.putText(frame, "DISTRACTED", (x_min, y_min - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:  # attentive
            color = (0, 255, 0)  # Green
        
        # Draw face bounding box with state color
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    def detect_banned_objects(self, frame):
        """Detect banned objects using YOLOv8"""
        try:
            results = self.yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id in self.banned_objects and confidence > 0.5:
                            object_name = self.banned_objects[class_id]
                            self.log_message(f"Warning: Banned object detected - {object_name}")
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{object_name}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            pass  # Silently handle object detection errors
    
    def count_persons(self, frame):
        """Count number of persons in frame"""
        try:
            results = self.yolo_model(frame, verbose=False)
            person_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id == 0 and confidence > 0.5:  # Person class
                            person_count += 1
            
            if person_count > 1:
                self.log_message(f"Warning: {person_count} persons detected - possible cheating")
                
        except Exception as e:
            pass  # Silently handle person counting errors
    
    def update_video_display(self, frame):
        """Update the video display in GUI"""
        try:
            # Resize frame for display
            display_frame = cv2.resize(frame, (640, 480))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep a reference
            
            # Update violation count
            self.violations_label.config(text=f"Violations: {len(self.violation_log)}")
            
        except Exception as e:
            pass  # Silently handle display errors
    
    def generate_report(self):
        """Generate detailed monitoring report"""
        try:
            if not self.violation_log and self.total_frames == 0:
                messagebox.showwarning("No Data", "No monitoring data available to generate report")
                return
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                title="Save Monitoring Report"
            )
            
            if not filename:
                return
            
            # Generate report content
            report_content = self.create_report_content()
            
            # Save report
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            messagebox.showinfo("Report Generated", f"Monitoring report saved to: {filename}")
            self.log_message(f"Report generated: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def create_report_content(self):
        """Create detailed report content"""
        report = []
        report.append("=" * 60)
        report.append("AI-POWERED EXAM PROCTORING SYSTEM - MONITORING REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Session Information
        report.append("SESSION INFORMATION:")
        report.append("-" * 30)
        if self.session_start_time:
            report.append(f"Session Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if not self.is_monitoring:
                duration = datetime.now() - self.session_start_time
                report.append(f"Session Duration: {str(duration).split('.')[0]}")
        report.append(f"Student Name: {self.student_name}")
        report.append(f"Student ID: {self.student_id}")
        report.append(f"Total Frames Processed: {self.total_frames}")
        report.append(f"Total Violations: {len(self.violation_log)}")
        report.append("")
        
        # Cognitive State Analysis
        if self.total_frames > 0:
            report.append("COGNITIVE STATE ANALYSIS:")
            report.append("-" * 30)
            attentive_pct = (self.cognitive_states['attentive'] / self.total_frames) * 100
            distracted_pct = (self.cognitive_states['distracted'] / self.total_frames) * 100
            drowsy_pct = (self.cognitive_states['drowsy'] / self.total_frames) * 100
            absent_pct = (self.cognitive_states['absent'] / self.total_frames) * 100
            
            report.append(f"Attentive: {attentive_pct:.1f}% ({self.cognitive_states['attentive']} frames)")
            report.append(f"Distracted: {distracted_pct:.1f}% ({self.cognitive_states['distracted']} frames)")
            report.append(f"Drowsy: {drowsy_pct:.1f}% ({self.cognitive_states['drowsy']} frames)")
            report.append(f"Absent/No Face: {absent_pct:.1f}% ({self.cognitive_states['absent']} frames)")
            report.append("")
        
        # Violation Log
        if self.violation_log:
            report.append("DETAILED VIOLATION LOG:")
            report.append("-" * 30)
            for i, violation in enumerate(self.violation_log, 1):
                timestamp = violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"{i:3d}. [{timestamp}] {violation['violation']}")
            report.append("")
        
        # Summary and Recommendations
        report.append("SUMMARY AND RECOMMENDATIONS:")
        report.append("-" * 30)
        violation_count = len(self.violation_log)
        
        if violation_count == 0:
            report.append("✓ No violations detected during the monitoring session.")
            report.append("✓ Student behavior appears to be compliant with exam protocols.")
        elif violation_count <= 3:
            report.append("⚠ Minor violations detected. Review recommended.")
            report.append("• Consider providing additional guidance on exam protocols.")
        else:
            report.append("⚠ Multiple violations detected. Detailed review required.")
            report.append("• Recommend manual review of exam session.")
            report.append("• Consider additional proctoring measures.")
        
        if self.total_frames > 0:
            if self.cognitive_states['absent'] / self.total_frames > 0.1:
                report.append("⚠ Student was frequently absent from frame.")
            if self.cognitive_states['drowsy'] / self.total_frames > 0.15:
                report.append("⚠ Student showed signs of drowsiness during exam.")
        
        report.append("")
        report.append("Report generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    root = tk.Tk()
    app = ExamProctoringSystem(root)
    
    def on_closing():
        if app.is_monitoring:
            app.stop_monitoring()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
    
    def update_video_display(self, frame):
        """Update the video display in GUI"""
        try:
            # Resize frame for display
            display_frame = cv2.resize(frame, (640, 480))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep a reference
            
            # Update violation count
            self.violations_label.config(text=f"Violations: {len(self.violation_log)}")
            
        except Exception as e:
            pass  # Silently handle display errors
    
    def generate_report(self):
        """Generate detailed monitoring report"""
        try:
            if not self.violation_log and self.total_frames == 0:
                messagebox.showwarning("No Data", "No monitoring data available to generate report")
                return
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                title="Save Monitoring Report"
            )
            
            if not filename:
                return
            
            # Generate report content
            report_content = self.create_report_content()
            
            # Save report
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            messagebox.showinfo("Report Generated", f"Monitoring report saved to: {filename}")
            self.log_message(f"Report generated: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def create_report_content(self):
        """Create detailed report content"""
        report = []
        report.append("=" * 60)
        report.append("AI-POWERED EXAM PROCTORING SYSTEM - MONITORING REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Session Information
        report.append("SESSION INFORMATION:")
        report.append("-" * 30)
        if self.session_start_time:
            report.append(f"Session Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if not self.is_monitoring:
                duration = datetime.now() - self.session_start_time
                report.append(f"Session Duration: {str(duration).split('.')[0]}")
        report.append(f"Student Name: {self.student_name}")
        report.append(f"Student ID: {self.student_id}")
        report.append(f"Total Frames Processed: {self.total_frames}")
        report.append(f"Total Violations: {len(self.violation_log)}")
        report.append("")
        
        # Cognitive State Analysis
        if self.total_frames > 0:
            report.append("COGNITIVE STATE ANALYSIS:")
            report.append("-" * 30)
            attentive_pct = (self.cognitive_states['attentive'] / self.total_frames) * 100
            distracted_pct = (self.cognitive_states['distracted'] / self.total_frames) * 100
            drowsy_pct = (self.cognitive_states['drowsy'] / self.total_frames) * 100
            absent_pct = (self.cognitive_states['absent'] / self.total_frames) * 100
            
            report.append(f"Attentive: {attentive_pct:.1f}% ({self.cognitive_states['attentive']} frames)")
            report.append(f"Distracted: {distracted_pct:.1f}% ({self.cognitive_states['distracted']} frames)")
            report.append(f"Drowsy: {drowsy_pct:.1f}% ({self.cognitive_states['drowsy']} frames)")
            report.append(f"Absent/No Face: {absent_pct:.1f}% ({self.cognitive_states['absent']} frames)")
            report.append("")
        
        # Violation Log
        if self.violation_log:
            report.append("DETAILED VIOLATION LOG:")
            report.append("-" * 30)
            for i, violation in enumerate(self.violation_log, 1):
                timestamp = violation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"{i:3d}. [{timestamp}] {violation['violation']}")
            report.append("")
        
        # Summary and Recommendations
        report.append("SUMMARY AND RECOMMENDATIONS:")
        report.append("-" * 30)
        violation_count = len(self.violation_log)
        
        if violation_count == 0:
            report.append("✓ No violations detected during the monitoring session.")
            report.append("✓ Student behavior appears to be compliant with exam protocols.")
        elif violation_count <= 3:
            report.append("⚠ Minor violations detected. Review recommended.")
            report.append("• Consider providing additional guidance on exam protocols.")
        else:
            report.append("⚠ Multiple violations detected. Detailed review required.")
            report.append("• Recommend manual review of exam session.")
            report.append("• Consider additional proctoring measures.")
        
        if self.total_frames > 0:
            if self.cognitive_states['absent'] / self.total_frames > 0.1:
                report.append("⚠ Student was frequently absent from frame.")
            if self.cognitive_states['drowsy'] / self.total_frames > 0.15:
                report.append("⚠ Student showed signs of drowsiness during exam.")
        
        report.append("")
        report.append("Report generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    root = tk.Tk()
    app = ExamProctoringSystem(root)
    
    def on_closing():
        if app.is_monitoring:
            app.stop_monitoring()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()