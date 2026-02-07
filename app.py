import streamlit as st
import subprocess
import sys

st.set_page_config(
    page_title="VisionPilot",
    layout="centered"
)

st.title("🛩️ VisionPilot – Gesture Controlled Drone UI")
st.markdown("""
Control a drone using **hand gestures** and **index-finger joystick**.

**Peace sign ✌️** → Toggle joystick mode  
""")

if st.button("🚀 Start Gesture Controller"):
    st.success("Starting VisionPilot...")
    subprocess.Popen([sys.executable, "main.py"])

st.warning("⚠️ Webcam opens in a separate OpenCV window")
