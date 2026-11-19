# AI-Based Plastic Waste Classification

This project is a responsive web application that identifies plastic waste categories using a trained deep learning model and the device camera or uploaded image.

## Features
- Camera access through the browser
- Image upload support
- Deep learning classification using MobileNetV2
- Plastic categories: PET, HDPE, LDPE, PP, PS, Other Plastic
- Confidence score and recycling guidance
- Responsive UI for desktop and mobile devices

## Tech Stack
- Python
- Flask
- TensorFlow/Keras
- MobileNetV2
- OpenCV
- NumPy
- HTML/CSS/JavaScript

## Project Structure
- `app.py` – Flask web application and prediction API
- `train_model.py` – model training script
- `templates/index.html` – responsive web UI
- `saved_models/` – saved model directory
- `data/` – folder for training and validation image datasets

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare dataset in the following structure:
   ```text
   data/
   ├── train/
   │   ├── PET/
   │   ├── HDPE/
   │   ├── LDPE/
   │   ├── PP/
   │   ├── PS/
   │   └── Other Plastic/
   └── val/
       ├── PET/
       ├── HDPE/
       ├── LDPE/
       ├── PP/
       ├── PS/
       └── Other Plastic/
   ```
4. Train the model:
   ```bash
   python train_model.py
   ```
5. Run the Flask app:
   ```bash
   python app.py
   ```
6. Open in browser:
   ```text
   http://localhost:5000
   ```

## Notes
- The model file is expected at `saved_models/plastic_classifier.keras`.
- If no model is available, the app returns a friendly message instead of misclassifying an image.
- Use a clear background and good lighting for better predictions.

## Future Improvements
- Real-time video classification
- Prediction history dashboard
- PWA support
- Edge AI / on-device inference
- IoT-based sorting integration
