# Cabbage Analyzer

A Python-based application with a GUI built using DearPyGui for analyzing cabbage images.  
This tool allows you to train a regression model from image datasets and predict cabbage shelf life using LAB color features and damaged area analysis.

---

## Key Features
- GUI for training and testing cabbage analysis models.
- Training pipeline with automatic background removal and LAB color extraction.
- Model saved in `.pkl` format with corresponding healthy color reference.
- Visualization of regression plots (L, a, b values, and damaged area).
- Prediction tab for testing images and estimating shelf life.
- Progress indicators for both training and testing processes.

---

## Technologies Used
- GUI: DearPyGui, Tkinter  
- Image Processing: OpenCV, rembg, Pillow  
- Data Handling: Pandas, NumPy  
- Modeling: scikit-learn (Linear Regression)  
- Visualization: Matplotlib  

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/username/cabbage-analyzer.git
```

### 2. Navigate to the project directory
```bash
cd cabbage-analyzer
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python cabbage_analyzer.py
```

---

## Notes
- Training data should be organized in folders named `Day 0`, `Day 1`, `Day 2`, etc., each containing cabbage images (`.JPG` format).
- The trained model will be saved in the chosen output folder with the name `<model_name>_model.pkl`.
- Healthy LAB color reference is also saved as `<model_name>_healthy_color.npy`.
- During testing, select a trained model and an image file (`.jpg`, `.jpeg`, `.png`) to perform predictions.
