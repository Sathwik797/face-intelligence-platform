# Face Recognition Attendance System

A Python-based face recognition system that automates attendance tracking using a webcam. This project uses OpenCV, face_recognition, and Flask to capture faces, recognize them, and log attendance in real time.

## 🚀 Features

- Real-time face recognition using webcam
- Automatic attendance logging to CSV file
- User-friendly web interface built with Flask
- Dataset collection and model training included

## 🧠 Tech Stack

- Python
- OpenCV
- face_recognition
- Flask
- HTML/CSS (Jinja2 templates)
- CSV for attendance logging

## 📁 Project Structure

face_recognition_attendence_system/ │ ├── app.py                 
# Flask app (main entry) ├── model.py               
# Face recognition & training logic ├── dataset/              
# Folder to store face images ├── trained_model/         
# Saved encodings for recognition ├── attendance.csv         
# Attendance log file ├── templates/            
# HTML files ├── static/                 # CSS/JS and image files └── requirements.txt        # Python dependencies

## 🛠 How to Run

1. *Clone the repo*
   ```bash
   git clone https://github.com/Sathwik797/face_recognition_attendence_system.git
   cd face_recognition_attendence_system

2. Install dependencies

pip install -r requirements.txt


3. Run the application

python app.py

4. Open in browser

http://127.0.0.1:5000/

📸 How It Works

Captures video from webcam

Detects and recognizes faces using a pre-trained model

Matches faces with stored encodings

Logs names and timestamps into attendance.csv


✍ Author

Sathwik Reddy – GitHub Profile
---

📌 Notes

Make sure your webcam is connected and accessible.

To add new users, place their images in the dataset/ folder and re-run the training using model.py.
