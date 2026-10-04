import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
from typing import Tuple, Optional
import logging

class EnhancedDrowsinessDetector:
    def __init__(self, landmark_path: str = 'shape_predictor_68_face_landmarks.dat'):
        """
        Initialize the drowsiness detector with improved parameters and features
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load detectors
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(landmark_path)
        except RuntimeError as e:
            self.logger.error(f"Error loading facial landmark predictor: {e}")
            raise
        
        # Enhanced eye landmark indices
        self.EYE_LANDMARKS = {
            'left': (42, 48),
            'right': (36, 42)
        }
        
        # Mouth landmarks
        self.MOUTH_LANDMARKS = (48, 68)
        
        # Improved detection parameters
        self.config = {
            'EYE_AR_THRESH': 0.25,  # Increased threshold for better sensitivity
            'EYE_AR_CONSEC_FRAMES': 20,  # Increased frame count for more reliable detection
            'YAWN_THRESH': 0.6,  # Normalized yawn threshold
            'BLINK_THRESH': 3,  # Blinks per minute threshold
            'DROWSY_TIME_THRESH': 1.5,  # Time threshold in seconds
        }
        
        # Enhanced tracking variables
        self.reset_tracking()

    def reset_tracking(self):
        """Reset all tracking variables"""
        self.drowsy_counter = 0
        self.blink_counter = 0
        self.yawn_counter = 0
        self.total_blinks = 0
        self.last_blink_time = time.time()
        self.start_time = time.time()

    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate enhanced Eye Aspect Ratio with error handling
        """
        try:
            # Vertical distances
            A = dist.euclidean(eye_points[1], eye_points[5])
            B = dist.euclidean(eye_points[2], eye_points[4])
            
            # Horizontal distance
            C = dist.euclidean(eye_points[0], eye_points[3])
            
            # Improved EAR calculation with validation
            if C == 0:
                return 0.0
                
            ear = (A + B) / (2.0 * C)
            return min(ear, 1.0)  # Normalize EAR
            
        except Exception as e:
            self.logger.warning(f"Error calculating EAR: {e}")
            return 0.0

    def calculate_mar(self, mouth_points: np.ndarray) -> float:
        """
        Calculate enhanced Mouth Aspect Ratio with normalization
        """
        try:
            # Vertical distances
            A = dist.euclidean(mouth_points[13], mouth_points[19])
            B = dist.euclidean(mouth_points[14], mouth_points[18])
            C = dist.euclidean(mouth_points[15], mouth_points[17])
            
            # Horizontal distance
            D = dist.euclidean(mouth_points[12], mouth_points[16])
            
            if D == 0:
                return 0.0
                
            mar = (A + B + C) / (3.0 * D)
            return min(mar, 1.0)  # Normalize MAR
            
        except Exception as e:
            self.logger.warning(f"Error calculating MAR: {e}")
            return 0.0

    def get_face_roi(self, frame: np.ndarray) -> Optional[dlib.rectangle]:
        """
        Get the region of interest (ROI) for the face with the highest confidence
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if not faces:
            return None
            
        # Return the largest face detected
        return max(faces, key=lambda rect: rect.width() * rect.height())

    def detect_drowsiness(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, dict]:
        """
        Enhanced drowsiness detection with multiple indicators
        """
        metrics = {
            'ear': 0.0,
            'mar': 0.0,
            'blink_rate': 0.0,
            'drowsy_time': 0.0
        }
        
        is_drowsy = False
        face_roi = self.get_face_roi(frame)
        
        if face_roi is None:
            return frame, is_drowsy, metrics
            
        # Get facial landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, face_roi)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        # Calculate eye metrics
        left_ear = self.calculate_ear(shape[self.EYE_LANDMARKS['left'][0]:self.EYE_LANDMARKS['left'][1]])
        right_ear = self.calculate_ear(shape[self.EYE_LANDMARKS['right'][0]:self.EYE_LANDMARKS['right'][1]])
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate mouth metrics
        mar = self.calculate_mar(shape[self.MOUTH_LANDMARKS[0]:self.MOUTH_LANDMARKS[1]])
        
        # Update metrics
        metrics.update({
            'ear': avg_ear,
            'mar': mar,
            'blink_rate': self.calculate_blink_rate(),
            'drowsy_time': self.drowsy_counter / 30.0  # Assuming 30 FPS
        })
        
        # Enhanced drowsiness detection logic
        if avg_ear < self.config['EYE_AR_THRESH']:
            self.drowsy_counter += 1
            
            if self.drowsy_counter >= self.config['EYE_AR_CONSEC_FRAMES']:
                is_drowsy = True
                self.draw_alert(frame, "DROWSINESS ALERT!", (10, 30))
        else:
            if self.drowsy_counter >= self.config['EYE_AR_CONSEC_FRAMES']:
                self.total_blinks += 1
                self.last_blink_time = time.time()
            self.drowsy_counter = 0
        
        # Yawn detection
        if mar > self.config['YAWN_THRESH']:
            self.yawn_counter += 1
            self.draw_alert(frame, "YAWNING DETECTED!", (10, 60))
            if self.yawn_counter > 20:  # Sustained yawning
                is_drowsy = True
        else:
            self.yawn_counter = max(0, self.yawn_counter - 1)
        
        # Draw facial landmarks and metrics
        self.draw_landmarks(frame, shape)
        self.draw_metrics(frame, metrics)
        
        return frame, is_drowsy, metrics

    def predict_from_image(self, image_path) -> Tuple[np.ndarray, bool, dict]:
        """
        Perform drowsiness detection on a single image
        """
        try:
            # frame = cv2.imread(image_path)
            # if frame is None:
            #     raise ValueError("Could not load image")
            
            # Reset tracking variables for single image analysis
            self.reset_tracking()
            
            # Perform detection
            return self.detect_drowsiness(image_path)
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None, False, {}

    def calculate_blink_rate(self) -> float:
        """
        Calculate blinks per minute
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0
        return (self.total_blinks * 60) / elapsed_time

    def draw_alert(self, frame: np.ndarray, text: str, position: Tuple[int, int]):
        """Draw alert message with enhanced visibility"""
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def draw_landmarks(self, frame: np.ndarray, shape: np.ndarray):
        """Draw facial landmarks with different colors for eyes and mouth"""
        # Draw eye contours
        for start, end in self.EYE_LANDMARKS.values():
            eye_points = shape[start:end]
            cv2.drawContours(frame, [cv2.convexHull(eye_points)], -1, (0, 255, 0), 1)
        
        # Draw mouth contours
        mouth_points = shape[self.MOUTH_LANDMARKS[0]:self.MOUTH_LANDMARKS[1]]
        cv2.drawContours(frame, [cv2.convexHull(mouth_points)], -1, (0, 255, 255), 1)

    def draw_metrics(self, frame: np.ndarray, metrics: dict):
        """Draw metrics on frame"""
        y_position = 30
        for key, value in metrics.items():
            cv2.putText(frame, f"{key.upper()}: {value:.2f}", 
                       (300, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 2)
            y_position += 30


    def process_video(self, video_path: str, output_path: Optional[str] = None) -> dict:
        """
        Process a video file for drowsiness detection with temporal smoothing
        and robust decision making.
        
        Args:
            video_path: Path to the input video file
            output_path: Optional path to save the processed video
            
        Returns:
            Dictionary containing analysis results and overall drowsiness assessment
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")
                
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer if output path is provided
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
            # Initialize tracking variables
            self.reset_tracking()
            drowsy_windows = []  # Track drowsiness in sliding windows
            window_size = fps * 5  # 5-second windows
            metrics_history = []
            frame_count = 0
            
            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break    
                # Process frame
                processed_frame, is_drowsy, metrics = self.detect_drowsiness(frame)
                frame_count += 1
                
                # Update tracking
                drowsy_windows.append(1 if is_drowsy else 0)
                if len(drowsy_windows) > window_size:
                    drowsy_windows.pop(0)
                    
                metrics_history.append(metrics)
                
                # Add progress indicator
                progress = (frame_count / total_frames) * 100
                cv2.putText(processed_frame, f"Progress: {progress:.1f}%", 
                        (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2)
                
                # Write frame if output path is provided
                if output_path:
                    out.write(processed_frame)
            
            # Calculate final results
            results = self.analyze_video_results(metrics_history, drowsy_windows, fps)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            raise
            
        finally:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            
    def analyze_video_results(self, metrics_history: list, drowsy_windows: list, fps: int) -> dict:
        """
        Analyze collected metrics and make final drowsiness assessment
        """
        if not metrics_history:
            return {}
            
        # Calculate average metrics
        avg_metrics = {
            'ear': np.mean([m['ear'] for m in metrics_history]),
            'mar': np.mean([m['mar'] for m in metrics_history]),
            'blink_rate': np.mean([m['blink_rate'] for m in metrics_history]),
            'drowsy_time': np.mean([m['drowsy_time'] for m in metrics_history])
        }
        
        # Calculate drowsiness percentage in sliding windows
        drowsy_percentage = (sum(drowsy_windows) / len(drowsy_windows)) * 100
        
        # Enhanced decision making with multiple factors
        drowsiness_score = 0
        
        # Factor 1: Average EAR vs threshold
        if avg_metrics['ear'] < self.config['EYE_AR_THRESH']:
            drowsiness_score += 35
            
        # Factor 2: Blink rate analysis
        if avg_metrics['blink_rate'] < self.config['BLINK_THRESH']:
            drowsiness_score += 25
            
        # Factor 3: Drowsy window percentage
        if drowsy_percentage > 30:  # If drowsy more than 30% of the time
            drowsiness_score += 40
            
        # Make final decision
        state = {
            'state': 'DROWSY' if drowsiness_score >= 60 else 'AWAKE',
            'confidence': drowsiness_score,
            'drowsy_percentage': drowsy_percentage,
            'average_metrics': avg_metrics,
            'analysis_summary': {
                'total_frames_analyzed': len(metrics_history),
                'video_duration': len(metrics_history) / fps,
                'ear_violations': sum(1 for m in metrics_history if m['ear'] < self.config['EYE_AR_THRESH']),
                'mar_violations': sum(1 for m in metrics_history if m['mar'] > self.config['YAWN_THRESH'])
            }
        }
        
        return state

def main():
    """Main function with enhanced error handling and user interface"""
    try:
        detector = EnhancedDrowsinessDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame, is_drowsy, metrics = detector.detect_drowsiness(frame)
            
            # Display frame
            cv2.imshow("Drowsiness Detection", frame)
            
            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
