# Habitat Suitability Modeling with Multi-Model Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red)](https://xgboost.readthedocs.io/)

A reproducible **presence/absence habitat-suitability** pipeline that
compares **Random Forest, XGBoost, Multilayer Perceptron (MLP)** and
**Logistic Regression** on multi-band environmental rasters, with
**SHAP** interpretability and full-raster suitability mapping.

## Features

- 📦 **Alignment-safe raster stacking** — all GeoTIFFs automatically
  reprojected/resampled to a common grid (`rasterio.warp.reproject`)
- 📍 **CRS-aware point extraction** — occurrence points (WGS84)
  are re-projected to the raster CRS before sampling
- 🔍 **Hyperparameter tuning** — 5-fold `GridSearchCV` on each model
- 📊 **Full evaluation suite** — ROC, AUC, Accuracy, Precision, Recall, F1
- 🗺️ **Suitability classification** — Low / Moderate / High per pixel
- 🔬 **SHAP for all 4 models** — tree, linear and kernel explainers
- 🧾 **Fully reproducible** — fixed seeds and saved model artifacts

## Dataset

Place your files in `data/raw/`:
