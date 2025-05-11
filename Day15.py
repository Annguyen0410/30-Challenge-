import cv2
import numpy as np
import math
from collections import deque

class HandGestureRecognizer:
    def __init__(self, smoothing_window_size=10):
        self.lower_skin_hsv = np.array([0, 48, 80], dtype=np.uint8)
        self.upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
        self.detection_history = deque(maxlen=smoothing_window_size)
        self.min_hand_contour_area = 5000
        self.defect_angle_threshold = 90 
        self.defect_depth_threshold = 20 
        self.fist_solidity_threshold = 0.85

    def _get_gesture_details(self, detection_code):
        if detection_code == -1: return 0, "No Hand"
        if detection_code == 0: return 0, "Fist"
        if detection_code == 1: return 1, "One"
        if detection_code == 2: return 2, "Two"
        if detection_code == 3: return 3, "Three"
        if detection_code == 4: return 4, "Four"
        if detection_code == 5: return 5, "Five / Open Palm"
        return 0, "Unknown"

    def detect(self, frame_input):
        frame = frame_input.copy()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv_frame, self.lower_skin_hsv, self.upper_skin_hsv)

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.erode(skin_mask, morph_kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, morph_kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_raw_detection_code = -1 

        if contours:
            hand_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(hand_contour) > self.min_hand_contour_area:
                hull_points = cv2.convexHull(hand_contour)
                cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
                cv2.drawContours(frame, [hull_points], -1, (0, 0, 255), 2)

                hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
                
                valid_defects_count = 0
                if hull_indices is not None and len(hull_indices) > 3 and len(hand_contour) > 3:
                    defects = cv2.convexityDefects(hand_contour, hull_indices)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d_val = defects[i, 0]
                            start_pt = tuple(hand_contour[s][0])
                            end_pt = tuple(hand_contour[e][0])
                            far_pt = tuple(hand_contour[f][0])

                            side_a = math.dist(start_pt, end_pt)
                            side_b = math.dist(start_pt, far_pt)
                            side_c = math.dist(end_pt, far_pt)
                            
                            denominator = 2 * side_b * side_c
                            if denominator == 0: continue
                            
                            angle_rad_numerator = (side_b**2 + side_c**2 - side_a**2) / denominator
                            angle_rad_numerator = max(min(angle_rad_numerator, 1.0), -1.0) 
                            
                            angle_rad = math.acos(angle_rad_numerator)
                            angle_deg = math.degrees(angle_rad)
                            defect_depth = d_val / 256.0

                            if angle_deg < self.defect_angle_threshold and defect_depth > self.defect_depth_threshold:
                                valid_defects_count += 1
                                cv2.circle(frame, far_pt, 5, [255, 0, 255], -1)
                
                if valid_defects_count == 0:
                    contour_area = cv2.contourArea(hand_contour)
                    hull_area = cv2.contourArea(hull_points)
                    if hull_area > 0:
                        solidity = contour_area / hull_area
                        if solidity > self.fist_solidity_threshold: current_raw_detection_code = 0
                        else: current_raw_detection_code = 1
                    else: current_raw_detection_code = 1 
                elif valid_defects_count == 1: current_raw_detection_code = 2
                elif valid_defects_count == 2: current_raw_detection_code = 3
                elif valid_defects_count == 3: current_raw_detection_code = 4
                elif valid_defects_count >= 4: current_raw_detection_code = 5
        
        self.detection_history.append(current_raw_detection_code)
        
        smoothed_detection_code = -1
        if self.detection_history:
            try:
                smoothed_detection_code = max(set(self.detection_history), key=list(self.detection_history).count)
            except ValueError:
                smoothed_detection_code = current_raw_detection_code

        fingers_to_display, gesture_text = self._get_gesture_details(smoothed_detection_code)

        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if gesture_text not in ["No Hand", "Unknown"]:
             cv2.putText(frame, f"Fingers: {fingers_to_display}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        return frame, skin_mask

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    recognizer = HandGestureRecognizer(smoothing_window_size=10)

    show_skin_mask = False 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        
        processed_frame, skin_mask_output = recognizer.detect(frame)

        if show_skin_mask:
            cv2.imshow("Skin Mask", skin_mask_output)
        
        cv2.imshow("Hand Gesture Recognition", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'): 
            show_skin_mask = not show_skin_mask
            if not show_skin_mask:
                cv2.destroyWindow("Skin Mask")


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()