# FaSIVA Quick Start & Testing Guide

## System Status ✅

All components trained, evaluated, and ready for testing:
- **Liveness Detection Accuracy:** 89.05%
- **All Models Loaded:** ✅ 
- **Database System:** ✅ Functional
- **Environment:** Python 3.11.6 with PyTorch 2.0.1

---

## Quick Start (5 Minutes)

### 1. Activate Virtual Environment
```bash
cd "/Users/admin/Documents/MSC_eng/First Semester/MCF685(Research Methodology)/Assignments/Fasiva implementation"
source .venv/bin/activate
```

### 2. Test the System
```bash
# Run interactive demo
python3 main.py
```

### 3. Test Individual Components
```bash
# Test face detection
python3 -c "
from main import FaSIVA
fasiva = FaSIVA()
result = fasiva.process_image('dataset/test/Andrej/1.jpeg')
if result['success']:
    print('✅ Face detected and processed')
    print(f'Face shape: {result[\"face_image\"].shape}')
else:
    print('❌ Failed to detect face')
"

# Test identification
python3 -c "
from main import FaSIVA
fasiva = FaSIVA()
result = fasiva.identify_person('dataset/test/Andrej/1.jpeg')
if result['identification']['success']:
    print('✅ Person identified')
    print(f'ID: {result[\"identification\"][\"person_id\"]}')
else:
    print('❌ No match found')
"
```

---

## Complete System Test Workflow

### Step 1: Register a New Person
```python
from main import FaSIVA
import glob

# Initialize system
fasiva = FaSIVA()

# Register person from multiple images
test_images = glob.glob('dataset/test/Andrej/*.jpeg')[:5]
result = fasiva.register_person(test_images, person_name="Test Person")

if result['success']:
    person_id = result['person_id']
    print(f"✅ Registered person with ID: {person_id}")
else:
    print(f"❌ Registration failed: {result}")
```

### Step 2: Identify the Person
```python
from main import FaSIVA

fasiva = FaSIVA()

# Try to identify someone
result = fasiva.identify_person('dataset/test/Andrej/2.jpeg')

if result['identification']['success']:
    print(f"✅ Identified: {result['identification']['person_name']}")
    print(f"   Distance: {result['identification']['distance']:.4f}")
else:
    print(f"❌ Could not identify person")
    print(f"   Min distance: {result['identification']['distance']:.4f}")
```

### Step 3: Verify Person Identity
```python
from main import FaSIVA

fasiva = FaSIVA()

# Verify specific person
person_id = 1  # ID from registration
result = fasiva.verify_person('dataset/test/Andrej/3.jpeg', person_id)

if result['verification']['verified']:
    print(f"✅ Person verified successfully")
    print(f"   Distance: {result['verification']['distance']:.4f}")
else:
    print(f"❌ Verification failed")
```

### Step 4: Complete Authentication (ID + Verify + Liveness)
```python
from main import FaSIVA

fasiva = FaSIVA()

# Full authentication
result = fasiva.authenticate_person('dataset/test/Andrej/4.jpeg')

if result['authentication']['success']:
    print(f"✅ AUTHENTICATION GRANTED")
    print(f"   Person: {result['authentication']['person_name']}")
    print(f"   Confidence: {result['authentication']['confidence']:.2%}")
    print(f"   Liveness Check: {result['authentication']['liveness_success']}")
else:
    print(f"❌ AUTHENTICATION DENIED")
    print(f"   Reason: {result['authentication']['stage_failed']}")
```

---

## Component Testing Examples

### Test Face Detection Module
```python
from face_detection import face_detector
from utils import load_image
import cv2

# Load image
image = load_image('dataset/test/Andrej/1.jpeg')

# Detect faces
faces, detections = face_detector(image)

print(f"Faces detected: {len(faces)}")
for i, face in enumerate(faces):
    print(f"  Face {i+1}: shape={face.shape}")

# Save detected faces
for i, face in enumerate(faces):
    cv2.imwrite(f"detected_face_{i}.jpg", face)
```

### Test Super-Resolution Module
```python
from super_resolution import get_fsrcnn_model, apply_super_resolution
from utils import load_image, get_resolution
import cv2

# Load model
fsrcnn = get_fsrcnn_model()

# Load low-res image
image = load_image('dataset/test/Andrej/1.jpeg')
print(f"Original resolution: {get_resolution(image)}")

# Apply super-resolution
if fsrcnn:
    enhanced = apply_super_resolution(image, fsrcnn)
    print(f"Enhanced resolution: {get_resolution(enhanced)}")
    cv2.imwrite("enhanced_face.jpg", enhanced)
else:
    print("FSRCNN model not available")
```

### Test Feature Extraction
```python
from feature_extraction import feature_extractor
from utils import load_image
import numpy as np

# Load face image
face = load_image('dataset/test/Andrej/1.jpeg')

# Extract identification vector (F vector - 2062D)
f_vector = feature_extractor.extract_f_vector(face)
print(f"F vector (Identification): shape={f_vector.shape}")
print(f"  Min: {f_vector.min():.4f}, Max: {f_vector.max():.4f}")

# Extract verification vector (E vector - 128D)
e_vector = feature_extractor.extract_e_vector(face)
print(f"E vector (Verification): shape={e_vector.shape}")
print(f"  Min: {e_vector.min():.4f}, Max: {e_vector.max():.4f}")

# Calculate similarity between two faces
face2 = load_image('dataset/test/Andrej/2.jpeg')
f_vector2 = feature_extractor.extract_f_vector(face2)
distance = np.linalg.norm(f_vector - f_vector2)
print(f"Distance between faces: {distance:.4f}")
```

### Test Liveness Detection
```python
from liveness_detection import liveness_detector
from utils import load_image

# Load face image
face = load_image('dataset/test/Andrej/1.jpeg')

# Get authentication vector
a_vector = liveness_detector.get_authentication_vector(face)
print(f"Authentication vector: {a_vector}")
print(f"  a1 (Reflection-based): {a_vector[0]}")
print(f"  a2 (Eye blink): {a_vector[1]}")

# Interpret results
if all(a_vector):
    print("✅ Face appears to be REAL (liveness passed)")
else:
    print("⚠️ Face may be SPOOFED (liveness check failed)")
```

---

## Batch Processing Example

```python
from main import FaSIVA
import glob

fasiva = FaSIVA()

# Process multiple images
test_images = glob.glob('dataset/test/**/*.jpeg', recursive=True)[:10]

print(f"Processing {len(test_images)} images...")

for img_path in test_images:
    result = fasiva.process_image(img_path)
    if result['success']:
        print(f"✅ {img_path}")
    else:
        print(f"❌ {img_path}: {result.get('error', 'Unknown error')}")
```

---

## Performance Evaluation

### Run Full System Evaluation
```bash
# Evaluate all trained models
python3 train.py --component evaluate
```

### Expected Output:
```
Evaluating Models...
==================================================
Evaluating ResNet-50 for identification...
- Dataset: LFW (13,233 images, 5,749 identities)
- Status: ✅ Complete

Evaluating CNNLivenessDetector...
- Dataset: NUAA + Replay-Attack + CASIA-2 (9,715 images)
- Accuracy: 89.05% ✅

Operation complete!
```

---

## Database Management

### Access Registered Persons
```python
from database import FaceDatabase

db = FaceDatabase('faces_database.db')

# Get all registered persons
cursor = db.conn.cursor()
cursor.execute("SELECT id, name FROM persons")
persons = cursor.fetchall()

print(f"Registered persons: {len(persons)}")
for person_id, name in persons:
    print(f"  - {name} (ID: {person_id})")

db.close()
```

### View Access Logs
```python
from database import FaceDatabase

db = FaceDatabase('faces_database.db')

# Get access logs
stats = db.get_statistics()
print(f"Total access attempts: {stats.get('total_access_attempts', 0)}")
print(f"Successful authentications: {stats.get('successful_accesses', 0)}")
print(f"Suspicious attempts: {stats.get('suspicious_accesses', 0)}")

db.close()
```

---

## Troubleshooting

### Issue: "No faces detected"
```python
# Check if image exists and is readable
from utils import load_image
import cv2

image = load_image('image_path.jpg')
if image is None:
    print("Image file not found or unreadable")
else:
    print(f"Image loaded: shape={image.shape}")
    # Try manual face detection
    from face_detection import face_detector
    faces, _ = face_detector(image)
    if len(faces) == 0:
        print("Face detection failed - image may not contain faces")
```

### Issue: "Module not found"
```bash
# Verify virtual environment is activated
which python3

# Install missing packages
pip install torch torchvision facenet-pytorch dlib opencv-python tqdm matplotlib scikit-learn
```

### Issue: Low identification accuracy
```python
# Check verification thresholds
from config import IDENTIFICATION_THRESHOLD, VERIFICATION_THRESHOLD
print(f"Identification threshold: {IDENTIFICATION_THRESHOLD}")
print(f"Verification threshold: {VERIFICATION_THRESHOLD}")

# Reduce thresholds for more lenient matching (but lower security)
# Or increase for stricter matching (but higher rejection rate)
```

---

## Performance Tips

### Optimize for Speed
```python
from main import FaSIVA

fasiva = FaSIVA()

# Batch process multiple images
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = fasiva.batch_process(images, mode='identify')

print(f"Processed {len(results)} images")
```

### Use GPU When Available
```python
import torch
from config import DEVICE

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name()}")
    print(f"Using device: {DEVICE}")
else:
    print("GPU not available, using CPU")
```

---

## System Statistics

```python
from main import FaSIVA

fasiva = FaSIVA()

# Get system statistics
stats = fasiva.get_statistics()
print(f"Processed images: {stats['system_stats'].get('processed_images', 0)}")
print(f"Successful identifications: {stats['system_stats'].get('successful_identifications', 0)}")
print(f"Spoofing attempts detected: {stats['system_stats'].get('spoofing_attempts', 0)}")
```

---

## Next Steps

1. ✅ Test each component individually using examples above
2. ✅ Register test persons and verify identification
3. ✅ Evaluate system on full test dataset
4. ✅ Generate performance metrics and plots
5. ✅ Document results for research submission
6. ✅ Prepare final research paper

---

**For detailed evaluation results, see `EVALUATION_RESULTS.md`**
