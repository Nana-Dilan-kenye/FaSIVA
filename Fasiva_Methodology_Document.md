# FaSIVA: Facial Signature for Identification, Verification and Authentication of Persons - Implemented Methodology

## 1. Introduction and FaSIVA Signature

The FaSIVA (Facial Signature for Identification, Verification and Authentication) system addresses challenges in face recognition, particularly concerning low-resolution images and spoofing attacks. It proposes a robust face representation, termed a "face signature," that integrates multiple parameters to enhance accuracy and reliability in real-life applications. The core idea is to move beyond simple identification by incorporating reinforcement processes such as image enhancement, verification, and authentication.

The FaSIVA signature, as formally defined in the paper, is represented as:

`S(I) = (R, F, E, A)`

Where:
*   `I`: Represents a facial image.
*   `R`: The super-resolution vector, indicating the quality (height `h` and width `w`) of the input image.
*   `F`: The identification vector, derived from patterns extracted from the image.
*   `E`: The verification vector, providing a secondary layer of pattern extraction for confirmation.
*   `A`: The authentication vector, which captures characteristics of human presence, specifically eye blinking and liveness detection.

The overall workflow of the FaSIVA system, as implemented, follows a sequential pipeline (inspired by Figure 1 in the paper):
1.  **Face Detection:** Identify and crop the face from the input image.
2.  **Resolution Check & Super-Resolution:** Assess image quality and enhance if necessary.
3.  **Feature Extraction:** Extract identification (F) and verification (E) vectors.
4.  **Authentication:** Generate the authentication (A) vector through liveness and blink detection.
5.  **Identification/Verification/Authentication:** Use the complete signature for person recognition and anti-spoofing.

## 2. Component Implementations

### 2.1. Face Detection
The initial step in the FaSIVA pipeline involves accurately detecting and aligning faces within an input image.
*   **Methodology:** The system utilizes the Multi-Task Cascaded Convolutional Networks (MTCNN) approach [31] for robust face detection and alignment. MTCNN is a cascade of three neural networks designed to detect and align faces within a bounding box.
*   **Implementation:** The `face_detection.py` module integrates `facenet_pytorch.MTCNN` to perform this task. It processes input images, identifies face bounding boxes, extracts face regions, and filters detections based on a confidence threshold (`FACE_DETECTION_CONFIDENCE` in `config.py`).

### 2.2. Super-Resolution
To address issues with low-resolution input images, FaSIVA incorporates a super-resolution module.
*   **Methodology:** The paper proposes using a convolutional neural network for super-resolution, specifically mentioning the Fast Super-Resolution Convolutional Neural Network (FSRCNN) [34]. This approach aims to reconstruct a higher-resolution image from a low-resolution input, enhancing hidden details crucial for subsequent feature extraction. The system applies super-resolution if the input image's resolution falls below a predefined threshold (`THRESHOLD_RESOLUTION`). The super-resolution factor (`k`) is set to 4. The model is trained specifically on face images using the Labeled Faces in the Wild (LFW) dataset.
*   **Implementation:** The `super_resolution.py` module implements the FSRCNN architecture. It includes the feature extraction, shrinking, mapping, expanding, and deconvolution layers as described in the paper. The `FaceSRDataset` class is designed to prepare training data from the LFW dataset, generating low-resolution and high-resolution patch pairs. The `apply_super_resolution` function handles the application of the trained FSRCNN model to enhance image quality.

### 2.3. Feature Extraction
Feature extraction is central to creating the `F` (identification) and `E` (verification) vectors of the FaSIVA signature.
*   **Methodology:**
    *   **Identification Vector (F):** Obtained using the ResNet-50 model [12], extracting features on an input image of size 224x224. The paper specifies this vector to be 2062 dimensions.
    *   **Verification Vector (E):** Obtained using the FaceNet model [9], which extracts characteristics using the "triplet loss" algorithm [35]. The paper specifies this vector to be 128 dimensions.
*   **Implementation:** The `feature_extraction.py` module handles both `F` and `E` vector extraction.
    *   **F Vector:** A ResNet-50 model (`torchvision.models.resnet50`) is used. It is initialized with ImageNet pre-trained weights, and a custom linear layer is added to output a 2062-dimensional feature vector (`RESNET_FEATURES_DIM`).
    *   **E Vector:** The `facenet_pytorch.InceptionResnetV1` model is used. This model typically outputs 512-dimensional embeddings. To align with the paper's specification of 128 dimensions, a linear layer (`nn.Linear(512, FACENET_FEATURES_DIM)`) has been added to reduce the output dimension to 128. Images are preprocessed to 160x160 and normalized to [-1, 1] as expected by FaceNet.

## 3. Authentication and Conclusion

### 3.1. Authentication Vector (A)
The authentication vector `A = (a1, a2)` is designed to verify human presence and prevent spoofing attacks.
*   **Methodology:**
    *   `a1` (Reflection-based Liveness): The paper describes calculating an acceptance coefficient based on the radius light incident on the facial surface (equation 7). This involves complex factors like `f_c(x,y)`, `ρ(x,y)`, `A_light`, and `cosθ`.
    *   `a2` (Eye Blink Detection): This component verifies liveness by detecting eye blinks. It uses the Eye Aspect Ratio (EAR) formula [36, equation 8, 15] based on facial landmarks. The implemented system uses a "proposed" method that considers each eye separately, detecting a blink if *any* eye's EAR falls below a threshold, which is more robust than requiring both eyes to blink simultaneously [paper, page 11].
    *   **CNN-based Liveness Detector:** An additional CNN-based liveness detector is employed to differentiate real individuals from spoofed ones (images or videos). The paper mentions an adapted AlexNet architecture [39, 40] trained on multiple datasets.
*   **Implementation:** The `liveness_detection.py` module implements these authentication mechanisms.
    *   **`a1` (Reflection-based Liveness):** The `ReflectionDetector` calculates a reflection coefficient. Instead of a direct implementation of the complex equation 7, a refined heuristic is used. This heuristic combines:
        *   Luminance (Y channel) entropy and variance.
        *   Chrominance (Cr, Cb channels) entropy.
        *   Gradient magnitude (edge information) from the Y channel.
        *   Image sharpness (Laplacian variance) from the Y channel.
        These features are weighted and combined to produce a coefficient indicative of liveness, aiming to capture the natural reflectance patterns of real faces.
    *   **`a2` (Eye Blink Detection):** The `EyeBlinkDetector` uses `dlib` for facial landmark detection. It calculates the EAR for both left and right eyes and implements the proposed single-eye blink detection logic.
    *   **CNN-based Liveness Detector:** The `CNNLivenessDetector` implements an adapted AlexNet model. It is designed to take 64x64 RGB images as input and classify them as real or fake. The model is trained using a combination of the NUAA, Replay-Attack, and CASIA datasets, as specified in the paper.

### 3.2. Conclusion
The implemented FaSIVA system provides a comprehensive solution for facial identification, verification, and authentication. By integrating super-resolution for image quality enhancement, robust feature extraction using ResNet-50 and FaceNet, and a multi-faceted liveness detection mechanism (reflection-based, eye blink, and CNN-based), the system aims to be accurate, efficient, and resilient against spoofing attacks in real-world scenarios. The modifications made to align the FaceNet output dimension and enhance the reflection-based liveness detection, along with the inclusion of all specified training datasets, further strengthen the system's adherence to the paper's methodology.

## References

[9] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering. In: IEEE conference on computer vision and pattern recognition. 6, (7):2015, p. 815–23. http://dx.doi.org/10.1109/cvpr.2015.7298682.
[12] Cao Q, Shen L, Xie W, Parkhi OM, Zisserman A. VGGFace2: A dataset for recognising faces across pose and age. In: IEEE conference on automatic face and gesture recognition. 2018, Cs.CV.
[31] Zhang K, Zhang Z, Li Z, Qiao Y. Joint face detection and alignment using multi-task cascaded convolutional networks. IEEE Signal Process Lett 2016. http://dx.doi.org/10.1109/LSP.2016.2603342.
[34] Soukupová T, Čech J. Real-Time Eye Blink Detection using Facial Landmark. In: 21st computer vision winter workshop. 2016.
[35] Schroff F, Kalenichenko D, Philbin J. Facenet: A unified embedding for face recognition and clustering. In: Computer vision and pattern recognition IEEE conference. 2015, http://dx.doi.org/10.1109/cvpr.2015.7298682.
[36] Sulayman N. Calculation of accuracy when designing measuring instruments. Meas Technol 2000;43(12):1031–4. http://dx.doi.org/10.1023/A:1010931516493.
[39] Ito K, Okano T, Aok T. Recent advances in biometric security:A case study of liveness detection in face recognition. In: Asia-Pacific signal and information processing association annual summit and conference. 12, (15):2017, p. 220–7. http://dx.doi.org/10.1109/APSIPA.2017.8282031.
[40] Koshy R, Mahmood A. Optimizing deep CNN architectures for FaceLiveness detection. Entropy 2019;21(423). http://dx.doi.org/10.3390/e21040423.
