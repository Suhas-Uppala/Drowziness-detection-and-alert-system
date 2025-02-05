# Drowsiness Detector

This project is a real-time drowsiness detection system using computer vision and facial landmarks. It uses OpenCV for video capture, MediaPipe for facial landmark detection, and Pygame for playing an alert sound when drowsiness is detected.

## Files

### face_active_detector1.py
This script captures video from the webcam, processes each frame to detect facial landmarks, and calculates the Eye Aspect Ratio (EAR) to determine drowsiness. If drowsiness is detected, an alert sound is played.

## Dependencies

To run this project, you need the following dependencies:

- OpenCV: `pip install opencv-python`
- MediaPipe: `pip install mediapipe`
- NumPy: `pip install numpy`
- SciPy: `pip install scipy`
- Pygame: `pip install pygame`

Make sure to install these dependencies before running the script.

## Usage

1. Ensure you have all the dependencies installed.
2. Place the alert sound file (`mixkit-alert-alarm-1005.wav`) in the same directory as the script.
3. Run the script using the command:
   ```bash
   python face_active_detector1.py
   ```
4. The script will start capturing video from your webcam and display two windows: one with the normal face view and another with the drowsiness detection view.
5. If drowsiness is detected, an alert sound will be played.

## License

This project is licensed under the MIT License.