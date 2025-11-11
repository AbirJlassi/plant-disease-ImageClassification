#  Plant Disease Classification

![Plant Disease Classification](assets/sample_images/demo_screenshot.png) <!-- optional image -->

This project classifies **38 plant diseases** from leaf images using deep learning models trained on the **PlantVillage dataset**. It provides an interactive **Streamlit interface** for uploading leaf images, comparing model performance, and visualizing training curves.

---

##  Project Overview

- **Goal:** Detect plant diseases from images of leaves.
- **Dataset:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes:** 38 plant disease categories
- **Image Size:** 224x224
- **Training Epochs:** 20
- **Batch Size:** 32

---

## Models Used

| Model | Architecture | Parameters | Best Val Accuracy | Best Val Loss | Precision | Recall |
|-------|-------------|------------|-----------------|---------------|-----------|--------|
| Baseline CNN | Custom 4-block CNN | 27,010,886 | 0.9803 | 0.0579 | 0.9825 | 0.9797 |
| MobileNetV2 Transfer | MobileNetV2 (ImageNet) + Custom Head | 3,058,022 (Trainable: 798,502) | 0.9504 | 0.1490 | 0.9591 | 0.9392 |

**Recommendation:** Baseline CNN performs better on this dataset.

---

## Screenshots

Below are some screenshots of the interface.
---
<img width="1891" height="853" alt="image" src="https://github.com/user-attachments/assets/5078710d-6859-4f6c-b497-4118d37a0143" />
<img width="1521" height="850" alt="image" src="https://github.com/user-attachments/assets/dda152b8-e5c8-414d-a67d-8803b9f5a1df" />
<img width="1491" height="610" alt="image" src="https://github.com/user-attachments/assets/a9704622-7e56-4bd1-83f5-90f37be239fa" />
<img width="1535" height="840" alt="image" src="https://github.com/user-attachments/assets/9bb1127a-310c-4ccf-9408-99b22d2319ac" />


##  Features

- Compare **Baseline CNN** and **MobileNetV2 Transfer Learning**.
- Visualize **accuracy** and **loss curves** during training.
- Upload leaf images and get **real-time predictions**.
- Display prediction **confidence** for each model.
- Interactive and user-friendly **Streamlit dashboard**.

---

##  Installation

1. Clone the repository:

```bash
git clone https://github.com/AbirJlassi/PlantDiseaseClassifier.git
cd PlantDiseaseClassifier
```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Run the App
```bash
streamlit run app.py
```
