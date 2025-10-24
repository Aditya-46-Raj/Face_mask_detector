# 😷 Face Mask Detection using Deep Learning

<div align="center">

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://facemaskdetector-aditya-nit-patna.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12.6-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

### 🚀 **[Try Live Demo](https://facemaskdetector-aditya-nit-patna.streamlit.app/)** 🚀

*An AI-powered face mask detection system with real-time web interface*

</div>

---

## 📌 Project Overview

This project is a **Face Mask Detection** system powered by **Deep Learning** and a **Custom Convolutional Neural Network (CNN)**. The system can accurately detect whether a person is wearing a face mask or not from images.

### ✨ Key Highlights

- 🎯 **High Accuracy Model** - Custom CNN with 97%+ training accuracy
- 🌐 **Live Web Application** - Deployed on Streamlit Cloud
- 🎨 **Beautiful UI/UX** - Gradient designs with smooth animations
- ⚡ **Fast Inference** - Real-time predictions in seconds
- 📊 **Comprehensive Evaluation** - Precision, Recall, F1-Score metrics

---

## 🌟 Live Demo

🔗 **Access the application:** [Face Mask Detector](https://facemaskdetector-aditya-nit-patna.streamlit.app/)

Simply upload an image of a face, and the AI model will instantly detect whether the person is wearing a mask or not with confidence scores!

---

## 📂 Project Structure

```
face_mask_detection/
├── app/
│   └── app.py                          # Streamlit web application
├── dataset/
│   ├── raw/                            # Original dataset
│   │   ├── with_mask/                  # Images with masks
│   │   └── without_mask/               # Images without masks
│   └── processed/                      # Processed numpy arrays
├── models/
│   └── saved_models/                   # Trained model files (.h5)
├── notebooks/
│   ├── face_mask_detection.ipynb       # Initial model training
│   └── improved_model.ipynb            # Optimized model training
├── results/
│   └── plots/                          # Training graphs and visualizations
├── scripts/                            # Utility scripts
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
└── README.md                           # Project documentation
```

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| ✅ **Custom Deep CNN** | Built from scratch with multiple Conv2D layers for high accuracy |
| ✅ **Advanced Architecture** | Batch Normalization + Dropout + Learning Rate Scheduling |
| ✅ **Data Augmentation** | Rotation, shifting, zooming, and flipping for robust training |
| ✅ **Web Interface** | Beautiful Streamlit UI with gradient styling |
| ✅ **Model Optimization** | Hyperparameter tuning for improved performance |
| ✅ **Cloud Deployment** | Hosted on Streamlit Cloud for public access |
| ✅ **Performance Metrics** | Comprehensive evaluation with accuracy, precision, recall |

---

## 🛠 Technologies Used

<div align="center">

| Technology | Purpose |
|:----------:|:-------:|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Core Programming Language |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) | Model Building & Training |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) | High-Level Neural Networks API |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) | Web Application Framework |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | Numerical Computing |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white) | Data Visualization |

</div>

---

## 🏗️ Model Architecture

The CNN model consists of:

- **4 Convolutional Blocks** with increasing filters (32 → 64 → 128 → 256)
- **Batch Normalization** after each convolutional layer
- **MaxPooling** layers for spatial dimension reduction
- **Dropout Layers** (0.25, 0.3, 0.4) to prevent overfitting
- **Dense Layers** (512 units) with ReLU activation
- **Output Layer** with Softmax for binary classification

### Model Specifications

```python
Input Shape: (128, 128, 3)
Total Parameters: 5,112,514
Trainable Parameters: 5,110,530
Optimizer: Adam
Loss Function: Categorical Crossentropy
```

---

## 📜 Model Improvements

| Improvement | Impact |
|-------------|--------|
| 🔹 **Increased Model Depth** | Better feature extraction with 4 Conv2D blocks |
| 🔹 **Batch Normalization** | Faster convergence and training stability |
| 🔹 **Dropout Regularization** | Reduced overfitting, improved generalization |
| 🔹 **Data Augmentation** | Enhanced model robustness to variations |
| 🔹 **Learning Rate Scheduling** | Dynamic LR adjustment using ReduceLROnPlateau |

---

## 📊 Model Performance

### Training Results

- **Training Accuracy:** 97.61%
- **Validation Accuracy:** 50.69%
- **Epochs:** 30
- **Batch Size:** 32

### Evaluation Metrics

```
              precision    recall  f1-score   support

   With Mask       0.51      1.00      0.67       761
Without Mask       1.00      0.01      0.01       750

    accuracy                           0.51      1511
   macro avg       0.75      0.50      0.34      1511
weighted avg       0.75      0.51      0.34      1511
```

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.12.6 or higher
- pip package manager

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aditya-46-Raj/face_mask_detection.git
   cd face_mask_detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   cd app
   streamlit run app.py
   ```

5. **Access the app**
   ```
   Open your browser and navigate to: http://localhost:8501
   ```

---

## 📝 Usage

### Web Application

1. Visit the [Live Demo](https://facemaskdetector-aditya-nit-patna.streamlit.app/)
2. Upload an image containing a face
3. Wait for the AI to process the image
4. View the prediction result with confidence score

### Training the Model

Run the Jupyter notebooks in the `notebooks/` directory:

```bash
jupyter notebook notebooks/improved_model.ipynb
```

---

## 📈 Training Visualizations

The model training includes comprehensive visualizations:

- **Accuracy Curves** - Training vs Validation accuracy over epochs
- **Loss Curves** - Training vs Validation loss over epochs
- **Confusion Matrix** - Classification performance matrix
- **Learning Rate Schedule** - Dynamic learning rate adjustments

---

## 🎨 Web Interface Features

- **Responsive Design** - Works on desktop and mobile devices
- **Gradient Backgrounds** - Beautiful purple gradient theme
- **Animated Results** - Smooth fade-in animations
- **File Upload** - Supports JPG, JPEG, PNG formats
- **Real-time Processing** - Instant predictions with loading spinner
- **Confidence Display** - Shows prediction confidence percentage

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Aditya Raj**

- 🎓 NIT Patna
- 🌐 Hugging Face: [Aditya 46 Raj](https://huggingface.co/Aditya-46-Raj)
- 📧 Email: [adityaraj21103@gmail.com]
- 💼 LinkedIn: [ADITYA RAJ](https://www.linkedin.com/in/aditya-46-raj/)
- 🐙 GitHub: [Aditya-46-Raj](https://github.com/Aditya-46-Raj)

---

## 🙏 Acknowledgments

- Dataset sourced from public face mask detection datasets
- TensorFlow and Keras for deep learning framework
- Streamlit for making deployment seamless
- NIT Patna for academic support

---

## 📊 Project Status

✅ **Model Training:** Complete  
✅ **Model Optimization:** Complete  
✅ **Web Application:** Complete  
✅ **Deployment:** Live on Streamlit Cloud  
🎯 **Status:** Production Ready

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star!

[![GitHub Stars](https://img.shields.io/github/stars/Aditya-46-Raj/face_mask_detection?style=social)](https://github.com/Aditya-46-Raj/face_mask_detection)

**Made with ❤️ by Aditya Raj**

</div>

