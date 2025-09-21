# Industrial Sound Event Detection and Anomaly Detection

This project implements a deep learning system for detecting sound events and anomalies in industrial machinery.  It combines event detection with machine condition monitoring using a semantic tree-based approach.
For comprehensive details, please refer to our paper: "A lightweight and interference-resilient sound anomaly detection model for industrial machines based on semantic embedding trees".


## Features

- **Event Detection**: The system can detect various sound events such as gear meshing, bearing faults, and other machinery sounds.
- **Anomaly Detection**: It can identify unusual patterns or deviations in the sound data, which may indicate potential issues with machinery.
- **Semantic Tree**: The system uses a semantic tree to represent the hierarchy of sound events, allowing for more nuanced and context-aware event detection.
- **Data Preprocessing**: The code includes functions for loading and preprocessing audio data, including feature extraction and normalization.


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/hackdong/adpaper-code-latest.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Download public dataset from MIMII, ESC-50-master, UrbanSound8k and organize it into the `dataset` directory as described in the structure. Run generate_synthetic_dataset_v3.py to generate synthetic dataset.

4. run the training script:trainv3.py

5. run validation script:valid.py

## Project Structure
├── dataset/ # Dataset directory

│ ├── MIMII/ # MIMII dataset

│ ├── ESC-50-master/ # ESC-50 dataset

│ ├── UrbanSound8k/ # UrbanSound8k dataset

│ ├── synthetic_dataset_v3/ # Training dataset

│ ├── synthetic_validation_dataset_v3/ # Validation dataset

│ ├── synthetic_test_dataset_v3/ # Test dataset

├── runs/ # Training runs and results

├── framework_visualization/ # Framework visualization outputs

├── generate_synthetic_dataset_v3.py

├── semantic_treev6.json # Semantic tree definition

├── semantictreemodelv3.py # Core model implementation

├── trainv3.py # Training script

├── test.py # Testing script

├── valid.py # Validation script

└── visuletestresult.py # Results visualization


