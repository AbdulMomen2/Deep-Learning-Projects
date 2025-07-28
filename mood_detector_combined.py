import cv2
from deepface import DeepFace
import serial
import time
import datetime
import csv

# Setup Arduino Serial
arduino = serial.Serial('COM5', 9600, timeout=3)  # Replace COM port as needed
time.sleep(2)

# Setup CSV logging
log_file = open("mood_log.csv", mode='a', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Detected Emotion", "Confidence"])

# Webcam
cap = cv2.VideoCapture(0)

# Map emotions to Arduino codes
def interpret_emotion(emotion):
    emotion = emotion.lower()
    if emotion in ['happy', 'surprise']:
        return 'H'
   
       
    elif emotion == 'sad':
        return 'S'
    elif emotion in ['angry', 'fear', 'disgust']:
        return 'A'
    else:
        return 'U'

# Track time to avoid spamming Arduino
last_sent = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Analyze emotions
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][dominant_emotion]

            # Show on screen
            label = f"{dominant_emotion} ({confidence:.1f}%)"
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Only send to Arduino every 3 seconds
            if time.time() - last_sent > 3:
                mood_code = interpret_emotion(dominant_emotion)
                print(f"Detected: {dominant_emotion}, Sending: {mood_code}")
                arduino.write(mood_code.encode())

                # Log
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([timestamp, dominant_emotion, f"{confidence:.1f}"])
                log_file.flush()

                last_sent = time.time()

        except Exception as e:
            print("Emotion detection failed:", e)
            cv2.putText(frame, "No face detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Mood Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    arduino.close()
    log_file.close()
    cv2.destroyAllWindows()
