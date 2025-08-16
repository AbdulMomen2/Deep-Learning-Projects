import serial
import time

arduino = serial.Serial('COM5', 9600)
time.sleep(2)  # Give time to Arduino

arduino.write(b'H')  # Try with 'H', 'N', 'S', or 'A'
print("Sent 'H'")
