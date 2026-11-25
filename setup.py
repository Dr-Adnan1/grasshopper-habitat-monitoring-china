from setuptools import setup, find_packages

setup(
    name="habitat-suitability",
    version="0.1.0",
    description="Multi-model habitat suitability modeling with SHAP",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy", "pandas", "scikit-learn", "xgboost",
        "rasterio", "pyproj", "matplotlib", "shap", "joblib",
        "openpyxl", "xlrd",
    ],
)