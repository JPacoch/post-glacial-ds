# Post-Glacial Landform Classification (Master's Thesis)

This repository contains the source code for a Master's thesis project focused on applying Computer Vision techniques to geomorphology. The project utilizes Convolutional Neural Networks (CNNs) to classify post-glacial landforms—specifically distinguishing between denuded and non-denuded features—using Hillshade raster data derived from Digital Elevation Models (DEM).

## Project Overview

The goal of this project is to automate the identification of specific geomorphological features using deep learning. The workflow involves:
1.  **Data Extraction**: Extracting image patches from Hillshade rasters based on labeled geospatial points.
2.  **Preprocessing**: Converting raster data into image formats suitable for CNN training.
3.  **Modeling**: Training a CNN to classify the landforms.
4.  **Evaluation**: Assessing model performance using ROC curves, confusion matrices, and accuracy metrics.
5.  **Visualization**: Visualizing feature maps to interpret the features learned by the convolutional layers.

## File Structure

*   **`data_processing.py`**: Handles the preparation of the dataset.
    *   Reads geospatial point data (GPKG) and Hillshade rasters (GeoTIFF).
    *   Clips 128x128 pixel windows around each point.
    *   Splits the data into training and testing sets.
    *   Saves the processed images into a directory structure compatible with Keras `ImageDataGenerator`.
*   **`training.py`**: The main script for training and evaluation.
    *   Loads the image datasets using `ImageDataGenerator`.
    *   Defines the CNN architecture (Conv2D, BatchNorm, MaxPooling, Dropout).
    *   Trains the model and saves the weights.
    *   Generates performance plots (ROC, Confusion Matrix).
    *   Visualizes intermediate layer activations (Feature Maps) for interpretability.
*   **`models_generators.py`**: Contains helper functions for model creation and training routines (`fitModel`).
*   **`plot.py`**: A utility class `PlotModel` for visualizing training history (accuracy/loss) and plotting feature maps.
*   **`config.py`**: (External dependency) Contains configuration constants such as file paths (`PATH_TO_HILLSHADE`, `PATH_TO_DENUDED`) and hyperparameters (`EPOCHS`, `BATCH_SIZE`).

## Dependencies

The project requires the following Python libraries:

*   `tensorflow`
*   `rasterio`
*   `geopandas`
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `scikit-learn`
*   `Pillow` (PIL)
*   `pyproj`

## Usage

1.  **Configuration**: Ensure a `config.py` file exists in the root directory with the necessary paths and hyperparameters defined.
2.  **Data Preparation**: Run the data processing script to generate the image dataset from the raw geospatial data.
    ```bash
    python data_processing.py
    ```
3.  **Training**: Run the training script to train the model and view evaluation metrics.
    ```bash
    python training.py
    ```