 Human Detection using Raspberry Pi & Thermal Imaging

 About
This project uses a Raspberry Pi with an MLX90640 thermal camera   to detect humans based on temperature patterns.
It processes real-time thermal frames, smooths readings, and predicts the presence of a human.

 Project Structure


├── main.py         # Main detection script
├── models/         # Trained model (.h5 file, not included)
├── data/           # Dataset (not included)
├── media/          # Photos and videos of setup & output
└── README.md

 Dataset
Captured using the MLX90640 thermal camera.

Contains thermal images of humans and non-humans.

Augmented with additional data from Kaggle.

Not included in this repository due to size constraints.

 Photos & Videos
Stored in the media/ folder.

Includes images of the hardware setup and example detection results.

Demonstrates how the system works in real-time.