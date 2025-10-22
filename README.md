# 2-up Time Trial Simulator

A simple Streamlit app to simulate a 2-up cycling time trial with configurable rider, course, and environment settings.  

## Features
- Adjustable rider mass, CdA, FTP, target front power, and pull duration (independent for Rider A and B).
- Adjustable air density, Crr, and placeholder wind/hill settings.
- Course length (out-and-back).
- Selectable plots:
  - Rolling 1-min Normalised Power
  - Cumulative NP since start
  - Cumulative average power since start

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py