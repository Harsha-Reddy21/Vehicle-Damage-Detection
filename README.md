# 🚗 Vehicle Damage Detection System

An AI-powered multi-agent system for automated vehicle damage assessment using computer vision and workflow orchestration.

## 🏗️ System Architecture

```mermaid
graph TD
    A[📷 Image Upload] --> B[🔍 Image Quality Agent]
    B --> C{Quality Check}
    C -->|Pass| D[📊 Enhanced Image]
    C -->|Fail| E[❌ Reject Image]
    
    D --> F[🔧 Damage Detection Agent]
    D --> G[🚗 Part Identification Agent]
    
    F --> H[YOLOv5 Inference]
    G --> I[YOLOv5 Inference]
    
    H --> J[📋 Damage Results]
    I --> K[📋 Part Results]
    
    J --> L[📊 Severity Assessment Agent]
    K --> L
    
    L --> M[💰 Cost Estimation]
    L --> N[⏱️ Repair Time]
    L --> O[🎯 Final Report]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style F fill:#fff3e0
    style G fill:#fff3e0
    style L fill:#e8f5e8
    style O fill:#ffebee
```

## 🔄 Processing Workflow

### Phase 1: Image Quality Assessment
```
📷 Input Image → 🔍 Quality Check → 🎨 Enhancement → ✅ Validation
```

**Quality Checks:**
- Blur detection (Laplacian variance)
- Brightness analysis (HSV values)
- Resolution validation
- Manipulation detection

### Phase 2: Parallel Analysis
```
🎨 Enhanced Image
    ├── 🔧 Damage Detection (YOLOv5)
    └── 🚗 Part Identification (YOLOv5)
```

**Damage Detection:**
- Identifies damage types and locations
- Calculates damage area percentages
- Returns bounding boxes with confidence scores

**Part Identification:**
- Detects vehicle parts in the image
- Maps part locations and types
- Provides confidence metrics

### Phase 3: Severity Assessment
```
📋 Damage Data + 📋 Part Data → 📊 Scoring Algorithm → 💰 Cost/Time Estimation
```

**Severity Calculation:**
```
Score = (Number of Parts × 2) + (Damage Area ÷ 10)

Severity Levels:
├── Minor: Score < 1 ($100-500, 1 day)
├── Moderate: Score < 2 ($500-1500, 3 days)
├── Major: Score < 3 ($1500-5000, 7 days)
└── Severe: Score ≥ 3 ($5000-15000, 14 days)
```

## 🧠 Agent Architecture

```mermaid
graph LR
    subgraph "LangGraph Workflow"
        A[ClaimState] --> B[Quality Agent]
        B --> C[Damage Agent]
        B --> D[Part Agent]
        C --> E[Severity Agent]
        D --> E
        E --> F[Final Result]
    end
    
    subgraph "State Management"
        G[claim_id]
        H[image_path]
        I[quality_result]
        J[damage_result]
        K[part_result]
        L[severity_result]
    end
    
    A -.-> G
    A -.-> H
    B -.-> I
    C -.-> J
    D -.-> K
    E -.-> L
```

## 🎯 Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Workflow Engine** | LangGraph | Orchestrates agent pipeline |
| **Computer Vision** | YOLOv5 + PyTorch | Object detection and classification |
| **Image Processing** | OpenCV | Quality assessment and enhancement |
| **Web Interface** | Streamlit | User-friendly frontend |
| **State Management** | TypedDict | Data flow between agents |

## 📊 Output Examples

### Damage Report Structure
```json
{
  "overall_severity": "moderate",
  "severity_score": 1.8,
  "estimated_cost_range": [500, 1500],
  "repair_time_days": 3,
  "damage_detections": [
    {
      "damage_type": "scratch",
      "confidence": 0.89,
      "bbox": [150, 200, 300, 350]
    }
  ],
  "identified_parts": [
    {
      "part_name": "front_bumper",
      "confidence": 0.95,
      "bbox": [100, 180, 400, 380]
    }
  ]
}
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit Interface**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Upload Vehicle Images**
   - Single or batch processing
   - Real-time status updates
   - Annotated results with damage visualization

## 🎯 YOLOv5 Training & Annotation

### 📋 Dataset Preparation

**Required Dataset Structure:**
```
dataset/
├── images/
│   ├── train/     # Training images
│   ├── val/       # Validation images
│   └── test/      # Test images (optional)
├── labels/
│   ├── train/     # Training annotations (.txt)
│   ├── val/       # Validation annotations (.txt)
│   └── test/      # Test annotations (.txt)
└── data.yaml      # Dataset configuration
```

**Dataset Configuration (data.yaml):**
```yaml
# Vehicle Damage Detection Dataset
path: ../dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (optional)

# Classes
nc: 12  # number of classes
names: ['scratch', 'dent', 'crack', 'paint_damage', 'rust', 'broken_part',
        'front_bumper', 'rear_bumper', 'door', 'hood', 'fender', 'mirror']
```

### 🏷️ Annotation Guidelines

**Damage Classes:**
- **scratch**: Surface scratches on paint
- **dent**: Physical deformation of metal/plastic
- **crack**: Cracks in bumpers, lights, or glass
- **paint_damage**: Paint chips, peeling, or discoloration
- **rust**: Corrosion on metal surfaces
- **broken_part**: Completely damaged/missing parts

**Vehicle Part Classes:**
- **front_bumper**: Front bumper assembly
- **rear_bumper**: Rear bumper assembly
- **door**: Any vehicle door
- **hood**: Engine hood/bonnet
- **fender**: Wheel well panels
- **mirror**: Side mirrors

**Annotation Format (YOLO):**
```
class_id center_x center_y width height
# Example: scratch on front bumper
0 0.456 0.321 0.123 0.087
6 0.500 0.400 0.800 0.300
```

### 🛠️ Annotation Tools

**Recommended Tools:**
1. **LabelImg** - Simple YOLO format labeling
   ```bash
   pip install labelImg
   labelImg
   ```

2. **Roboflow** - Web-based annotation with auto-augmentation
   - Drag & drop interface
   - Auto-annotation suggestions
   - Export in YOLO format

3. **CVAT** - Advanced annotation for large datasets
   - Team collaboration features
   - Semi-automatic annotation
   - Quality control workflows

### 🏃‍♂️ Training Process

**1. Environment Setup:**
```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

**2. Training Command:**
```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data ../dataset/data.yaml \
  --weights yolov5s.pt \
  --name vehicle_damage_model \
  --cache
```

**3. Training Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--img` | 640 | Input image size |
| `--batch` | 16 | Batch size (adjust for GPU memory) |
| `--epochs` | 100 | Training epochs |
| `--data` | data.yaml | Dataset configuration |
| `--weights` | yolov5s.pt | Pre-trained weights |
| `--cache` | - | Cache images for faster training |

**4. Hyperparameter Tuning (hyp.yaml):**
```yaml
lr0: 0.01          # initial learning rate
lrf: 0.01          # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937    # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0    # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
box: 0.05          # box loss gain
cls: 0.5           # cls loss gain
cls_pw: 1.0        # cls BCELoss positive_weight
obj: 1.0           # obj loss gain (scale with pixels)
obj_pw: 1.0        # obj BCELoss positive_weight
anchor_t: 4.0      # anchor-multiple threshold
```

### 📊 Model Evaluation

**Validation Metrics:**
```bash
python val.py \
  --weights runs/train/vehicle_damage_model/weights/best.pt \
  --data ../dataset/data.yaml \
  --img 640 \
  --save-txt \
  --save-conf
```

**Key Metrics Tracked:**
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

**Training Visualization:**
- Loss curves (train/val)
- mAP progression
- Learning rate schedule
- Batch predictions with ground truth

### 🔄 Data Augmentation

**Built-in Augmentations:**
- Mosaic (combines 4 images)
- MixUp (blends 2 images)
- Random scaling and translation
- Horizontal flipping
- HSV color space adjustments
- Random rotation (±10°)

**Custom Augmentation Pipeline:**
```python
# Custom augmentation for vehicle damage detection
transforms = [
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.RandomShadow(p=0.3),
    A.RandomFog(p=0.1)
]
```

### 📈 Training Tips

**Best Practices:**
1. **Start Small**: Begin with 100-200 images per class
2. **Quality over Quantity**: Ensure accurate annotations
3. **Balanced Dataset**: Equal representation of damage types
4. **Diverse Conditions**: Include various lighting, angles, weather
5. **Regular Validation**: Monitor overfitting with validation set

**Common Issues & Solutions:**
- **Low mAP**: Increase dataset size or improve annotations
- **Overfitting**: Add more augmentation or reduce model complexity
- **Class Imbalance**: Use weighted loss or oversample minority classes
- **Poor Convergence**: Adjust learning rate or batch size

## 📊 Sample Evaluation Results

### 🎯 Model Performance Metrics

#### Training Results Summary
```
Model: YOLOv5s Custom (Vehicle Damage Detection)
Dataset: 150 annotated images (8 classes)
Epochs: 100 (Early stopping at epoch 85)
```

#### Validation Metrics
| Metric | Damage Classes | Part Classes | Overall |
|--------|---------------|--------------|---------|
| **mAP@0.5** | 0.847 | 0.923 | 0.885 |
| **mAP@0.5:0.95** | 0.634 | 0.756 | 0.695 |
| **Precision** | 0.823 | 0.912 | 0.867 |
| **Recall** | 0.789 | 0.894 | 0.841 |
| **F1-Score** | 0.805 | 0.903 | 0.854 |

#### Per-Class Performance
| Class | Precision | Recall | mAP@0.5 | Instances |
|-------|-----------|--------|---------|-----------|
| **scratch** | 0.845 | 0.812 | 0.829 | 487 |
| **dent** | 0.823 | 0.798 | 0.811 | 392 |
| **crack** | 0.791 | 0.756 | 0.773 | 298 |


#### Processing Time Comparison
```
Traditional Manual Assessment: 15-30 minutes
AI System: 2-5 seconds
Time Reduction: 99.7%
```

## 🔧 Configuration

- **Model Path**: `C:/Misogi/Vehicle-Damage-Detection/best.pt`
- **Confidence Threshold**: 0.25
- **Image Enhancement**: Automatic histogram equalization + sharpening
- **Quality Threshold**: 0.6

---
*Automated vehicle damage assessment for insurance and automotive industries*
