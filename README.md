# **Speech Command Classification: Attacks & Defenses**

This project explores the vulnerability of Speech Command recognition models to adversarial attacks and implements defense mechanisms to improve robustness. It demonstrates how adversarial perturbations can fool a Convolutional Neural Network (CNN) trained on audio data and how specific defenses can mitigate these threats.

## **Project Configuration**

* **Application:** Speech Command Classification (35 classes)  
* **Attacks:** DeepFool, Carlini & Wagner (C\&W)  
* **Defenses:** Feature Squeezing, Adversarial Training

## **Project Structure**

The project is organized as follows:

* data/: Stores the Speech Commands dataset (automatically downloaded).  
* models/: Contains the model architecture and saved weights.  
  * simple\_cnn.py: The CNN architecture definition.  
  * baseline\_model.pth: Weights for the standard trained model.  
  * robust\_model.pth: Weights for the adversarially trained model.  
* attacks/: Custom implementations of adversarial attacks.  
  * deepfool.py: Implementation of the DeepFool iterative attack.  
  * cw.py: Implementation of the Carlini & Wagner L2 optimization attack.  
* defenses/: Custom implementations of defense mechanisms.  
  * squeezing.py: Feature Squeezing logic (Bit-depth reduction).  
* utils.py: Utility functions for data loading, preprocessing (MFCCs), and handling audio backends.  
* train.py: Script to train the baseline classification model on clean data.  
* train\_adv.py: Script to perform adversarial training (generating attacks on the fly and retraining).  
* test\_attacks.py: Script to generate attacks on the baseline model and visualize the results (original vs. adversarial spectrograms).  
* test\_defenses.py: Script to quantitatively verify the effectiveness of Feature Squeezing and the Robust Model against attacks.

## **Setup & Installation**

1. **Prerequisites:** Ensure you have Python 3.8+ installed.  
2. **Create** a Virtual **Environment (Recommended):**  
   python \-m venv .venv  
   \# Windows  
   .venv\\Scripts\\activate  
   \# Mac/Linux  
   source .venv/bin/activate

3. **Install Dependencies:**  
   pip install torch torchaudio matplotlib tqdm soundfile

## **Usage Guide**

### **1\. Train Baseline Model**

First, train the standard CNN on the Speech Commands dataset. This establishes the baseline accuracy.

python train.py

*Output: Saves the trained model to models/baseline\_model.pth.*

### **2\. Test Attacks**

Run the implemented attacks (DeepFool and C\&W) on a few samples from the test set to visualize how they affect the model.

python test\_attacks.py

*Output:* Displays MFCC plots of *the original audio, the adversarial audio, and the perturbation noise.*

### **3\. Train Robust Model (Defense 1: Adversarial Training)**

Perform adversarial training to create a more robust model. This script generates DeepFool attacks on the fly during training and teaches the model to classify them correctly.

python train\_adv.py

*Output:* Saves the robust model to *models/robust\_model.pth.*

### **4\. Evaluate Defenses**

Run the final evaluation script to test both defenses against fresh attacks.

python test\_defenses.py

*Output:* Prints the results of attacking a sample and then checking if (A) Feature Squeezing restores the correct label and (B) the Robust Model classifies *it correctly.*

## **Methodology & Results**

### **Baseline Performance**

The simple CNN achieves a reasonable accuracy on the 35-class Speech Commands dataset, establishing a baseline for standard audio classification.

### **Adversarial Attacks**

* **DeepFool:** Successfully finds the minimal perturbation required to cross the decision boundary. It is efficient and highly effective at inducing misclassification (e.g., changing "Yes" to "No").  
* **Carlini & Wagner (C\&W):** A powerful optimization-based attack. While computationally expensive (requiring many iterations), it generates high-confidence adversarial examples with minimal perceptible noise.

### **Defenses**

1. **Feature Squeezing:** Implemented via bit-depth reduction on the MFCC features. This simple pre-processing step successfully "sanitizes" many adversarial examples by destroying the precise noise patterns required to fool the model.  
2. **Adversarial Training:** Retraining the model on adversarial examples significantly improved its robustness. The robust\_model.pth is able to correctly classify adversarial audio that fooled the baseline model.

## **Comparison**

* **Attacks:** DeepFool proved to be faster for on-the-fly generation during training compared to C\&W.  
* **Defenses:** Feature Squeezing is a low-cost, inference-time defense that works surprisingly well. Adversarial Training provides inherent model