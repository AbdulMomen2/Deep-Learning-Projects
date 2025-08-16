import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import serial
import time
import datetime
import csv

# Setup Arduino Serial
arduino = serial.Serial('COM5', 9600)  # Change COM port if needed
time.sleep(2)

# Setup CSV logging
log_file = open("mood_log.csv", mode='a', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Detected Emotion"])

# Webcam
cap = cv2.VideoCapture(0)

# Map emotions to Arduino codes
def interpret_emotion(emotion):
    emotion = emotion.lower()
    if emotion in ['happy', 'surprise']:
        return 'H'  # For Happy/Surprise
    elif emotion == 'neutral':
        return 'N'  # For Neutral
    elif emotion == 'sad':
        return 'S'  # For Sad
    elif emotion in ['angry', 'fear', 'disgust']:
        return 'A'  # For Angry/Fear/Disgust
    else:
        return 'U'  # For Uncertain or unknown emotions

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            print("Detected Emotion:", dominant_emotion)

            # Send to Arduino
            mood_code = interpret_emotion(dominant_emotion)
            arduino.write(mood_code.encode())

            # Log mood
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([timestamp, dominant_emotion])
            log_file.flush()

            # Delay to avoid spamming
            time.sleep(3)

        except Exception as e:
            print("Emotion detection failed:", e)

        cv2.imshow("Mood Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    log_file.close()
    arduino.close()
    cv2.destroyAllWindows()
