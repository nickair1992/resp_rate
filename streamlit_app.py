import cv2
import numpy as np
import streamlit as st
import time
from collections import deque
import matplotlib.pyplot as plt

# Initialize Streamlit app
st.title("Respiratory Rate (RR) Monitoring App")

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Allow camera access to stream video.
2. Ensure your chest is visible in the video frame.
3. Observe the real-time respiratory rate and trend graph.
""")

# Streamlit components
placeholder_video = st.empty()
placeholder_rr = st.empty()
placeholder_graph = st.empty()

# Settings for respiratory rate estimation
frame_rate = 30  # Webcam frame rate (fps)
window_seconds = 30  # Analysis window for respiratory rate
roi_x, roi_y, roi_w, roi_h = 100, 200, 300, 100  # Region of Interest (ROI) in the frame

# Data for trend graph
rr_trend = deque(maxlen=100)

# Define a function for respiratory rate calculation
def estimate_respiratory_rate(signal, fps):
    fft_result = np.fft.rfft(signal - np.mean(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    magnitudes = np.abs(fft_result)
    respiratory_peak_idx = np.argmax(magnitudes[1:]) + 1  # Ignore DC component
    respiratory_rate = frequencies[respiratory_peak_idx] * 60  # Convert to breaths per minute
    return respiratory_rate

# Initialize variables
signal = deque(maxlen=frame_rate * window_seconds)
timestamps = deque(maxlen=frame_rate * window_seconds)
start_time = time.time()

# Access webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Webcam not accessible. Please ensure it is connected and accessible.")
    st.stop()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam feed. Restart the application.")
            break

        # Convert to grayscale for analysis
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Define ROI and visualize it
        roi = gray_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        avg_intensity = np.mean(roi)
        signal.append(avg_intensity)
        timestamps.append(time.time() - start_time)

        # Draw ROI on frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

        # Display video feed
        placeholder_video.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Calculate respiratory rate if enough data is collected
        if len(signal) > frame_rate:
            respiratory_rate = estimate_respiratory_rate(signal, frame_rate)
            rr_trend.append(respiratory_rate)

            # Display respiratory rate
            placeholder_rr.metric("Current Respiratory Rate (Breaths per Minute)", f"{respiratory_rate:.2f}")

            # Plot trend graph
            plt.figure(figsize=(10, 4))
            plt.plot(rr_trend, label="Respiratory Rate Trend")
            plt.xlabel("Time (s)")
            plt.ylabel("Respiratory Rate (Breaths per Minute)")
            plt.title("Respiratory Rate Trend Over Time")
            plt.legend()
            placeholder_graph.pyplot(plt)
            plt.close()

        # Exit condition for demo purposes
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except Exception as e:
    st.error(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
