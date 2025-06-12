# Monitoring_Image_Transformation
Advanced Monitoring image Transformation

# Advanced Transformation-Invariant MNIST Classification

## 🔍 Project Overview
This project systematically evaluates how **rotational discontinuities** and **combined transformations** affect MNIST digit classification, extending beyond standard rotation studies with two groundbreaking paradigms:

1. **Disjoint Rotation Intervals**: Models trained on non-overlapping angular ranges (e.g., `[0°,30°]` + `[120°,150°]`) to test generalization in gaps  
2. **Composite Transformations**: Joint rotation-translation augmentation to analyze transformation compositionality  

**Key Insight**: Reveals how neural networks learn (or fail to learn) geometric invariances, with implications for EEG signal augmentation via spatial-temporal transformations.

## 📊 Key Results
### Task 1: Disjoint Rotation Intervals
| Training Ranges          | Peak Accuracy | Gap Performance Drop |
|--------------------------|---------------|----------------------|
| `[0°,10°]` + `[90°,100°]` | 92%           | 34% at 50°           |
| `[20°,30°]` + `[70°,80°]` | 88%           | 41% at 50°           |
| `[40°,50°]` + `[60°,70°]` | 85%           | <5% (adjacent)       |

**Finding**: Models exhibit catastrophic performance drops in angular gaps >20° (Figure 1).

### Task 2: Composite Rotation+Translation
| Training Configuration          | Rotation-Only Test | Translation-Only Test |
|---------------------------------|--------------------|-----------------------|
| `[0°,25°]` + `[0,7px]`         | 89%                | 82%                   |
| `[0°,50°]` + `[0,15px]`        | 84%                | 78%                   |
| `[0°,100°]` + `[0,28px]`       | 76%                | 91%                   |

**Finding**: Translation training enhances rotation robustness at wide angles (Figure 3-4).

## 🎯 Extended Objectives
1. **Gap Analysis**: Quantify generalization in untrained angular ranges
2. **Transformation Disentanglement**: Determine if models learn rotations/translations as independent features
3. **EEG Adaptation Protocol**: Develop spatial-temporal augmentation for neural signals

## 🛠️ Enhanced Methodology
### Disjoint Rotations
```python
def generate_disjoint_rotations(intervals=[[0,30], [120,150]]):
    angle = np.random.choice([
        np.random.uniform(intervals[0][0], intervals[0][1]),
        np.random.uniform(intervals[1][0], intervals[1][1])
    ])
    return rotate_image(image, angle)
