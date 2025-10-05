# 🌌 A World Away — Hunting for Exoplanets Using AI

## 🚀 What It Does & How It Works

### 🧩 The Problem  
Identifying exoplanets from astronomical data requires extensive expertise and time-consuming analysis. Traditional methods involve manually examining light curves and stellar data.

### 💡 Our Solution  
#### Data Input  
Users provide astronomical parameters (either manually via sliders or via CSV upload).  

#### AI Classification  
A machine learning model analyzes features to predict **planet vs. non-planet**.  

#### Confidence Assessment  
The system provides **probability distributions** showing prediction certainty.  

#### Visualization  
For predicted planets, the system generates **interactive 3D models** based on physical properties.  

#### Habitability Analysis  
Evaluates factors such as **distance from star, temperature, and size** to assess life-supporting potential.  

#### Explainability  
**SHAP analysis** reveals which features influenced the classification decision.  

---

## ⚙️ Core Functionality  

Our exoplanet classification system uses a **Stacking Ensemble** combining **Random Forest**, **Gradient Boosting**, and **AdaBoost** classifiers.  
Trained on data from **NASA's Kepler, K2, and TESS** missions, it achieves:  
- **96.6% average confidence**  
- **92% of predictions** at high confidence levels (≥90%)  

It provides two prediction modes:

1. **Single Prediction**  
   - Real-time rendering of how the predicted planet would appear based on its physical characteristics.  
   - As parameters are adjusted, the system performs classification, calculates a **habitability score**, and generates a **3D simulation**.

2. **Batch Classification Mode**  
   - Reads a CSV file and automatically detects dataset type (**Kepler, K2, or TESS**).  
   - Detects exoplanets and displays **statistical analyses** with metrics including accuracy and precision.  
   - Visual performance evaluation through **ROC curves**, **Precision-Recall curves**, and **model confidence distributions**.  
   - **SHAP library** provides detailed feature importance for interpretability.

---

## 🎯 Benefits  

### ⏱️ Accelerates Research  
- Processes hundreds of candidates in seconds instead of hours.  
- Identifies promising targets for follow-up observations.  
- Reduces false positives through statistical validation.  

### 🎓 Educational Excellence  
- Interactive learning: see immediate effects of changing parameters.  
- Transparent AI demystifies machine learning for students.  
- Real satellite data connects classrooms to active space missions.  
- 3D visualization makes abstract concepts tangible.  

### 🌍 Accessibility  
- Runs directly in the **web browser** — no specialized software required.  
- Intuitive interface for non-astronomers.  
- Free access democratizes space research.  
- Supports standard data formats (**CSV/TSV**).  

### 🔬 Scientific Rigor  
- Multiple validation metrics ensure model reliability.  
- **SHAP explanations** make results interpretable for peer review.  
- Mission-specific auto-detection handles different detection methods.  
- Habitability scores grounded in established **astrobiology** principles.  

---

## 🌠 Intended Impact  

### 📘 Immediate Impact  
- **Education:** Inspire the next generation of astronomers and data scientists.  
- **Citizen Science:** Empower the public to participate in exoplanet discovery.  
- **Research Support:** Provide a free tool for preliminary candidate screening.  
- **Scientific Literacy:** Enhance public understanding of AI and astronomy.  

### 🔭 Long-term Vision  
- Contribute to the catalog of confirmed exoplanets.  
- Support the search for potentially habitable worlds.  
- Demonstrate **responsible AI** with built-in explainability.  
- Bridge the gap between professional astronomy and public engagement.  
- Foster interdisciplinary thinking (**Astronomy + AI + Visualization**).  

---

## 🛠️ Tools, Technologies, and Implementation  

### 🤖 Machine Learning Stack  
- **Python** — Model development and training  
- **Scikit-learn** — Classification algorithms  
- **SHAP** — Model interpretability and feature importance  
- **Pandas / NumPy** — Data processing and manipulation  

### 🌐 Web Development  
- **Frontend:** React.js  
- **Backend:** FastAPI (for model serving)  

### 🪐 3D Graphics  
- **Three.js** — WebGL-based 3D rendering engine  

### 📊 Data Visualization  
- **Matplotlib / Seaborn** — ROC, Precision-Recall, and Confidence curves  
- **SHAP Visualizations** — Feature importance plots  

---

## 💡 Creativity and Innovation  
- Unique **3D visualization** of predicted exoplanets.  
- Algorithm compatible with multiple file formats.  
- Trained on **NASA datasets** from Kepler, K2, and TESS missions.  
- Fully **explainable AI** via SHAP interpretability tools.  

---

## 🌌 Why This Matters  

The search for exoplanets is one of humanity’s most profound scientific endeavors — seeking to answer **whether we’re alone in the universe**.  
**A World Away: Hunting for Exoplanets Using AI** makes this cutting-edge research **accessible, engaging, and educational**.  

By combining **AI classification**, **interactive visualization**, and **explainable analytics**, we’re not just building a tool — we’re creating an **experience** that inspires curiosity about distant worlds.  

Our project shows that:  
- Complex science can be both **rigorous and accessible**.  
- AI can be **powerful and transparent**.  
- The hunt for planets light-years away can feel **immediate and personal** through thoughtful design.  

---

✨ *“Exploring distant worlds, powered by AI — because discovery should belong to everyone.”*
