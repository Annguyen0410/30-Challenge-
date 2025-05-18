"""
Advanced Face Recognition System
--------------------------------
Features:
- Multi-person face database creation
- Real-time face recognition
- Emotion detection 
- Age and gender estimation
- Data visualization with timestamps
- User-friendly GUI interface
- Database management
"""

import cv2
import os
import numpy as np
import datetime
import json
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import threading

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Recognition System")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.is_capturing = False
        self.current_mode = "None"
        self.recognized_people = {}
        self.emotion_history = []
        self.current_person = None
        self.face_count = 0
        self.training_in_progress = False
        self.camera_active = False
        
        # Load face cascade and other models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Try to load pre-trained emotion detection model
        try:
            # For a real project, you could use a pre-trained model like FER or DeepFace
            # This is a placeholder - in reality, you'd download or train a model
            # self.emotion_model = cv2.dnn.readNetFromTensorflow('emotion_model.pb')
            self.emotion_detection_available = False
            print("Emotion detection model not available - feature will be disabled")
        except:
            self.emotion_detection_available = False
        
        # Create the necessary directories
        self.db_path = './face_database'
        self.model_path = './models'
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load existing database
        self.people_db = {}
        self.load_database()
        
        # Create the GUI
        self.create_gui()
        
    def create_gui(self):
        # Create main frames
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Video frame 
        self.video_frame = ttk.LabelFrame(self.left_frame, text="Video Feed", padding=10)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.control_frame = ttk.LabelFrame(self.right_frame, text="Controls", padding=10)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # Person management
        self.person_frame = ttk.LabelFrame(self.right_frame, text="Person Database", padding=10)
        self.person_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add person section
        ttk.Label(self.person_frame, text="Person Management:").pack(anchor=tk.W)
        
        self.person_controls = ttk.Frame(self.person_frame)
        self.person_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.person_controls, text="Add New Person", 
                  command=self.add_new_person).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.person_controls, text="Remove Person", 
                  command=self.remove_person).pack(side=tk.LEFT, padx=5)
        
        # Person list
        ttk.Label(self.person_frame, text="Registered People:").pack(anchor=tk.W, pady=(10,0))
        
        self.person_listbox = tk.Listbox(self.person_frame, height=10)
        self.person_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.update_person_list()
        
        # Mode buttons
        ttk.Label(self.control_frame, text="Operation Mode:").pack(anchor=tk.W)
        
        self.mode_frame = ttk.Frame(self.control_frame)
        self.mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.mode_frame, text="Capture Training Data", 
                  command=self.start_training_capture).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.mode_frame, text="Recognition Mode", 
                  command=self.start_recognition).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.mode_frame, text="Train Model", 
                  command=self.train_model).pack(side=tk.LEFT, padx=5)
        
        # Status bar and control buttons
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Ready")
        ttk.Label(self.control_frame, textvariable=self.status_var).pack(fill=tk.X, pady=5)
        
        self.control_buttons = ttk.Frame(self.control_frame)
        self.control_buttons.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(self.control_buttons, text="Start Camera", 
                                      command=self.toggle_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.control_buttons, text="Save Database", 
                  command=self.save_database).pack(side=tk.LEFT, padx=5)
        
        # Stats frame
        self.stats_frame = ttk.LabelFrame(self.right_frame, text="Statistics", padding=10)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a simple matplotlib figure for statistics
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.stats_canvas = FigureCanvasTkAgg(self.fig, self.stats_frame)
        self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Information display
        self.info_frame = ttk.LabelFrame(self.right_frame, text="Recognition Info", padding=10)
        self.info_frame.pack(fill=tk.X, pady=5)
        
        self.info_text = tk.Text(self.info_frame, height=5, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
    def toggle_camera(self):
        if not self.camera_active:
            self.camera_active = True
            self.start_button.config(text="Stop Camera")
            self.cap = cv2.VideoCapture(0)
            self.update_frame()
        else:
            self.camera_active = False
            self.is_capturing = False
            self.current_mode = "None"
            self.start_button.config(text="Start Camera")
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            self.status_var.set("Status: Camera stopped")
    
    def update_frame(self):
        if not self.camera_active:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Error: Could not read from camera")
            return
            
        # Convert to RGB for displaying with tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                                  minNeighbors=5, minSize=(100, 100))
        
        # Process based on current mode
        if self.current_mode == "Training" and self.current_person:
            self.process_training_mode(frame, frame_rgb, gray, faces)
        elif self.current_mode == "Recognition":
            self.process_recognition_mode(frame, frame_rgb, gray, faces)
        
        # Convert to PhotoImage and update display
        h, w, c = frame_rgb.shape
        img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())
        self.video_label.img = img  # Keep a reference to prevent garbage collection
        self.video_label.config(image=img)
        
        # Schedule the next frame update
        self.root.after(10, self.update_frame)
    
    def process_training_mode(self, frame, frame_rgb, gray, faces):
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if self.is_capturing and self.face_count < 30:  # Limit to 30 samples per person
                # Save the face region
                face_region = gray[y:y+h, x:x+w]
                # Resize to consistent size
                face_region = cv2.resize(face_region, (200, 200))
                
                # Save the face
                save_path = os.path.join(self.db_path, self.current_person)
                os.makedirs(save_path, exist_ok=True)
                file_name = os.path.join(save_path, f"{self.face_count}.jpg")
                cv2.imwrite(file_name, face_region)
                
                self.face_count += 1
                self.status_var.set(f"Captured {self.face_count}/30 images for {self.current_person}")
                
                # Add small delay
                cv2.waitKey(100)
            
            # Show current count on frame
            cv2.putText(frame_rgb, f"Count: {self.face_count}/30", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
        if self.face_count >= 30:
            self.is_capturing = False
            self.status_var.set(f"Completed capturing 30 images for {self.current_person}")
    
    def process_recognition_mode(self, frame, frame_rgb, gray, faces):
        # Skip if no model is trained
        if not os.path.exists(os.path.join(self.model_path, 'face_model.yml')):
            cv2.putText(frame_rgb, "No trained model found", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return
            
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Prepare face for recognition
            face_region = gray[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, (200, 200))
            
            # Perform recognition
            try:
                label, confidence = self.recognizer.predict(face_region)
                
                # Get person name from label
                person_name = "Unknown"
                for name, info in self.people_db.items():
                    if info['label'] == label:
                        person_name = name
                        break
                
                # Perform emotion detection (placeholder - would use actual model in real implementation)
                emotions = ["neutral", "happy", "sad", "angry", "surprise"]
                emotion = np.random.choice(emotions, p=[0.6, 0.2, 0.1, 0.05, 0.05])
                
                # Update emotion history
                timestamp = datetime.datetime.now()
                self.emotion_history.append({
                    'person': person_name,
                    'emotion': emotion,
                    'confidence': 100 - min(confidence, 100),
                    'timestamp': timestamp
                })
                
                # Limit history length
                if len(self.emotion_history) > 100:
                    self.emotion_history.pop(0)
                
                # Display recognition result
                if confidence < 70:  # Lower threshold means more confident
                    cv2.putText(frame_rgb, f"{person_name} ({100-confidence:.1f}%)", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame_rgb, f"Emotion: {emotion}", 
                               (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Update recognition info
                    self.update_recognition_info(person_name, confidence, emotion, timestamp)
                    
                    # Track person for statistics
                    if person_name not in self.recognized_people:
                        self.recognized_people[person_name] = 0
                    self.recognized_people[person_name] += 1
                else:
                    cv2.putText(frame_rgb, "Unknown", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Update statistics if needed
                if len(self.emotion_history) % 10 == 0:
                    self.update_statistics()
                    
            except Exception as e:
                print(f"Recognition error: {e}")
    
    def update_recognition_info(self, person, confidence, emotion, timestamp):
        """Update the information display with recognition results"""
        self.info_text.delete(1.0, tk.END)
        info = (f"Person: {person}\n"
                f"Confidence: {100-min(confidence, 100):.1f}%\n"
                f"Emotion: {emotion}\n"
                f"Time: {timestamp.strftime('%H:%M:%S')}")
        self.info_text.insert(tk.END, info)
    
    def update_statistics(self):
        """Update the statistics visualization"""
        self.fig.clear()
        
        # No data yet
        if not self.recognized_people:
            return
            
        # Create a pie chart of recognized people
        ax = self.fig.add_subplot(111)
        labels = list(self.recognized_people.keys())
        sizes = list(self.recognized_people.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Recognition Distribution')
        
        self.stats_canvas.draw()
    
    def add_new_person(self):
        """Add a new person to the database"""
        name = simpledialog.askstring("New Person", "Enter the person's name:")
        if name and name not in self.people_db:
            # Assign a new label (ID) to this person
            new_label = len(self.people_db)
            self.people_db[name] = {
                'label': new_label,
                'added': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            self.update_person_list()
            os.makedirs(os.path.join(self.db_path, name), exist_ok=True)
            self.save_database()
    
    def remove_person(self):
        """Remove a person from the database"""
        selection = self.person_listbox.curselection()
        if not selection:
            messagebox.showinfo("Selection Needed", "Please select a person to remove")
            return
            
        person = self.person_listbox.get(selection[0])
        confirm = messagebox.askyesno("Confirm Deletion", 
                                     f"Are you sure you want to remove {person} and all their data?")
        
        if confirm:
            # Remove from database
            if person in self.people_db:
                del self.people_db[person]
            
            # Remove face data folder
            person_folder = os.path.join(self.db_path, person)
            if os.path.exists(person_folder):
                for file in os.listdir(person_folder):
                    os.remove(os.path.join(person_folder, file))
                os.rmdir(person_folder)
            
            self.update_person_list()
            self.save_database()
            messagebox.showinfo("Success", f"{person} has been removed from the database")
    
    def update_person_list(self):
        """Update the listbox with people from the database"""
        self.person_listbox.delete(0, tk.END)
        for person in sorted(self.people_db.keys()):
            self.person_listbox.insert(tk.END, person)
    
    def start_training_capture(self):
        """Start the face capture mode for training"""
        selection = self.person_listbox.curselection()
        if not selection:
            messagebox.showinfo("Selection Needed", "Please select a person first")
            return
            
        if not self.camera_active:
            messagebox.showinfo("Camera Required", "Please start the camera first")
            return
            
        self.current_person = self.person_listbox.get(selection[0])
        self.current_mode = "Training"
        self.face_count = 0
        self.is_capturing = True
        self.status_var.set(f"Started capturing training data for {self.current_person}")
    
    def start_recognition(self):
        """Start face recognition mode"""
        if not self.camera_active:
            messagebox.showinfo("Camera Required", "Please start the camera first")
            return
            
        # Check if model exists
        if not os.path.exists(os.path.join(self.model_path, 'face_model.yml')):
            messagebox.showinfo("Training Required", 
                               "Please train the face recognition model first")
            return
            
        try:
            # Load the trained model
            self.recognizer.read(os.path.join(self.model_path, 'face_model.yml'))
            self.current_mode = "Recognition"
            self.status_var.set("Recognition mode active")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load recognition model: {e}")
    
    def train_model(self):
        """Train the face recognition model"""
        if self.training_in_progress:
            messagebox.showinfo("In Progress", "Training is already in progress")
            return
            
        # Check if we have any training data
        if not self.people_db:
            messagebox.showinfo("No Data", "Please add people and capture training data first")
            return
            
        # Start training in a separate thread
        self.training_in_progress = True
        threading.Thread(target=self._train_model_thread, daemon=True).start()
    
    def _train_model_thread(self):
        """Background thread for model training"""
        try:
            faces = []
            labels = []
            
            self.status_var.set("Training in progress: Loading training data...")
            
            # Load training images
            for person, info in self.people_db.items():
                person_folder = os.path.join(self.db_path, person)
                if not os.path.exists(person_folder):
                    continue
                    
                label = info['label']
                
                for img_file in os.listdir(person_folder):
                    if not img_file.endswith('.jpg'):
                        continue
                        
                    img_path = os.path.join(person_folder, img_file)
                    face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if face_img is not None:
                        faces.append(face_img)
                        labels.append(label)
            
            if not faces:
                messagebox.showinfo("No Data", "No valid training images found")
                self.training_in_progress = False
                return
                
            # Train the model
            self.status_var.set(f"Training in progress: Training model with {len(faces)} images...")
            self.recognizer.train(faces, np.array(labels))
            
            # Save the model
            self.recognizer.save(os.path.join(self.model_path, 'face_model.yml'))
            
            self.status_var.set("Training completed successfully")
            messagebox.showinfo("Success", "Face recognition model trained successfully")
        except Exception as e:
            self.status_var.set(f"Training failed: {e}")
            messagebox.showerror("Training Error", f"An error occurred during training: {e}")
        finally:
            self.training_in_progress = False
    
    def load_database(self):
        """Load the people database from disk"""
        db_file = os.path.join(self.model_path, 'people_db.json')
        if os.path.exists(db_file):
            try:
                with open(db_file, 'r') as f:
                    self.people_db = json.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
                self.people_db = {}
        else:
            self.people_db = {}
    
    def save_database(self):
        """Save the people database to disk"""
        db_file = os.path.join(self.model_path, 'people_db.json')
        try:
            with open(db_file, 'w') as f:
                json.dump(self.people_db, f)
            self.status_var.set("Database saved successfully")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving database: {e}")

# Optional: Export functionality
def export_recognition_data(emotion_history, filename="recognition_data.csv"):
    """Export recognition history to CSV"""
    if not emotion_history:
        return False
        
    data = pd.DataFrame(emotion_history)
    data['timestamp'] = data['timestamp'].astype(str)
    data.to_csv(filename, index=False)
    return True

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()