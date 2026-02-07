# aims_drone_project
✋ VisionPilot — Hand Gesture Controlled Drone System

VisionPilot is a real-time hand gesture–controlled drone interface built using MediaPipe and lightweight neural networks, designed for accuracy, low latency, and intuitive human–drone interaction.

The project focuses on robust gesture recognition, mode-based control, and a virtual joystick system, making it suitable for academic submissions, demos, and future drone integrations.
🚀 Key Features

MediaPipe Hand Tracking
Uses MediaPipe Hands only for landmark extraction, ensuring fast and reliable hand detection.

ANN-Based Gesture Classification
A lightweight Artificial Neural Network (ANN) classifies gestures using 21 hand landmarks (63 features) instead of raw images, drastically reducing computation.

Dead Class for Safety
Includes a dedicated DEAD / NO-GESTURE class to prevent accidental commands when the hand is not in a valid gesture state.

Region-of-Interest (ROI) Control Box
Commands are executed only when the hand is inside a face-aligned control box, reducing false triggers and improving usability.

Gesture Hold Timer
Every command is executed only after holding a gesture for a fixed duration, increasing stability and safety.

Joystick Mode (Peace Gesture Activated)
A peace sign toggles a virtual joystick mode, where the index finger acts as a continuous joystick for smooth directional control and speed variation.

Speed-Based Control
Joystick speed increases linearly as the finger moves toward the edge of the control box (up to 100%).

Single-Hand Optimized
The system is optimized for one-hand operation to ensure consistent tracking and predictable behavior.