# NVIDIA MONAI Workshop 1

This repository contains a series of hands-on tutorials for the NVIDIA MONAI Workshop, focusing on medical image analysis using MONAI and PyTorch.

## 1. MONAI Label with OHIF Viewer

**Notebook:** `lab_1_MONAI_Label_with_OHIF_Viewer/lab_1_monailabel_radiology_OHIF.ipynb`

This tutorial demonstrates the end-to-end setup of the web-based OHIF viewer and MONAI Label plugin for radiology applications.

### Key Components:
- **App:** MONAI Label Radiology App
- **Viewer:** OHIF Viewer
- **Server:** Orthanc DICOM Server
- **Task:** Interactive Segmentation

### Workflow:
1.  **Environment Setup:** Install MONAI Label and dependencies.
2.  **App Download:** Download the standard "radiology" app.
3.  **Data Management:** Download Spleen CT DICOM data and upload it to the Orthanc DICOM server.
4.  **Server Launch:** Start the MONAI Label server connected to Orthanc.
5.  **Interactive Segmentation:** Use the OHIF viewer to perform auto-segmentation on the spleen using pre-trained models provided by MONAI Label.

## 2. 2D Diabetic Retinopathy Classification

**Notebook:** `lab_2_Classification_2D/lab_2_2d_classification_dr.ipynb`

This notebook demonstrates a 2D classification task to grade Diabetic Retinopathy (DR) from fundus images into five stages.

### Key Components:
- **Model:** `DenseNet121`
- **Frameworks:** MONAI, PyTorch Lightning
- **Task:** Multi-class classification
- **Loss Function:** `CrossEntropyLoss`

### Workflow:
1.  **Environment Setup:** Installs necessary libraries like MONAI and PyTorch Lightning.
2.  **Data Handling:** Downloads and extracts a dataset of DR images. It then dynamically creates file lists for training, validation, and test sets.
3.  **Data Augmentation:** Uses MONAI transforms (`LoadImageD`, `ScaleIntensityD`, `ResizeD`, `RandZoomD`) to prepare the data for the model.
4.  **Training:** A `pl.LightningModule` is defined to encapsulate the DenseNet121 model, optimizer, and training/validation/test steps. The training is managed by a PyTorch Lightning `Trainer`, utilizing callbacks for model checkpointing and early stopping, and logging results to TensorBoard.
5.  **Inference:** After training, the notebook shows how to load the best-saved model checkpoint to perform inference on a single test image and visualize the predicted grade.
6.  **Monitoring:** Includes checks for GPU VRAM usage before and after training.

## 3. 3D Spleen Segmentation from CT Scans

**Notebook:** `lab_3_Spleen_Segmentation_3D/lab_3_spleen_segmentation_3d.ipynb`

This tutorial covers a 3D volumetric segmentation task to delineate the spleen from abdominal CT scans, using the Medical Segmentation Decathlon (MSD) dataset.

### Key Components:
- **Model:** `3D UNet`
- **Frameworks:** MONAI, PyTorch
- **Task:** Volumetric semantic segmentation
- **Loss Function:** `DiceLoss`
- **Metric:** `DiceMetric`

### Workflow:
1.  **Data Handling:** Downloads and prepares the MSD Spleen dataset. It uses `CacheDataset` to accelerate data loading during training.
2.  **Advanced Transforms:** A robust pipeline of MONAI transforms is used for pre-processing and augmentation, including spacing correction, intensity scaling, foreground cropping, and balanced patch sampling (`RandCropByPosNegLabeld`).
3.  **Training Loop:** Implements a standard PyTorch training loop, evaluating the model every two epochs using `sliding_window_inference` for robust validation on the full-volume images.
4.  **Evaluation:** The model's performance is tracked using the `DiceMetric`. The best model is saved based on the highest validation Dice score.
5.  **Analysis:** Plots the training loss and validation Dice metric over epochs to visualize the learning progress.
6.  **Inference:** Demonstrates how to load the best model to run inference on validation images and visualize the input, ground truth, and predicted segmentation side-by-side. It also includes a section for running inference on a separate test set.

## 4. MONAI Label with Self-Trained Model

**Notebook:** `lab_4_MONAI_LABEL_selfmodel_OHIF/lab_4_monailabel_radiology_self_model_OHIF.ipynb`

This notebook shows how to integrate your own custom-trained model (from Lab 3) into the MONAI Label ecosystem.

### Key Components:
- **App:** MONAI Label Radiology App (Customized)
- **Model:** Custom trained 3D UNet (from Lab 3)
- **Integration:** Custom Inference Class

### Workflow:
1.  **Setup:** Prepare the MONAI Label environment and download the radiology app.
2.  **Customization:** Create a custom configuration file (`trained_spleen_segmentation.py`) to define the inference logic for the self-trained model.
3.  **Deployment:** Start the MONAI Label server, passing the custom configuration and the path to the best checkpoint saved in Lab 3.
4.  **Validation:** Access the OHIF viewer to verify that the self-trained model appears in the plugin and can successfully segment new images.

## 5. 3D Brain Tumor Segmentation with Swin UNETR

**Notebook:** `lab_5_Brain_Tumor_Segmentation_3D/lab_5_swin_unetr_brats21_segmentation_3d.ipynb`

This notebook tackles the challenging task of multi-class brain tumor segmentation from 3D multi-modal MRI scans, using data from the BraTS 21 challenge. It segments three tumor sub-regions: Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT).

### Key Components:
- **Model:** `SwinUNETR` (Swin Transformers for Semantic Segmentation)
- **Frameworks:** MONAI, PyTorch
- **Task:** Multi-class volumetric semantic segmentation
- **Loss Function:** `DiceLoss`
- **Metric:** `DiceMetric`

### Workflow:
1.  **Data Management:** Downloads the BraTS 21 dataset and uses a JSON file to organize the data into training and validation folds.
2.  **Specialized Transforms:** The transform pipeline is tailored for the BraTS dataset, featuring `ConvertToMultiChannelBasedOnBratsClassesd` to handle the specific label structure (NCR, ED, ET).
3.  **State-of-the-Art Model:** Utilizes the `SwinUNETR` model, a powerful transformer-based architecture for 3D medical image segmentation. Gradient checkpointing is enabled for memory-efficient training.
4.  **Structured Training:** A comprehensive training script is built with helper classes (`AverageMeter`) and functions (`train_epoch`, `val_epoch`, `trainer`) to manage the training and validation process.
5.  **Inference and Evaluation:** Uses `sliding_window_inference` for validation. The performance is evaluated for each tumor sub-region (TC, WT, ET) individually.
6.  **Results Visualization:** Plots the training loss and the Dice scores for each of the three sub-regions, providing a detailed view of the model's performance.
