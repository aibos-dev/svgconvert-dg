# svgconvert-dg
convert the generated  files into a more practical SVG format.

# Overview
AI-generated line art illustrations require conversion into vector graphics to allow designers to make manual edits. This involves transforming the raw images into formats compatible with illustration software, such as EPS files. The process includes binarization, skeletonization, vectorization, and final output as an EPS file readable by Adobe Illustrator.

In this module we make use of topological skeleton as a set of polylines from binary images as our input to obtain practical SVGs

# characteristics of practical SVGs
1. Path parameters are classified into reusable styles (e.g., CSS classes).
2. The number of paths is reduced.

## Introduction
This project requires Python version 3.11 or higher

This repository is actively under development and provides a tool to convert topological skeleton as a set of polylines from binary images into SVG format using a Python-based module. 

## Prerequisites
### 1. Install Docker
Ensure Docker and Docker Compose are installed on your system.
```bash
sudo apt install docker docker-compose
```

### 2. Install Poetry
Poetry is used for dependency management. Install it with:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Verify installation:
```bash
poetry --version
```
Install dependencies
```bash
poetry install
```
## File Structure
```
svg_converter/
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── LICENSE                 # License information
├── poetry.lock             # Dependency lock file
├── pyproject.toml          # Project configuration file
├── README.md               # Project documentation
├── src/
│   ├── converter.py        # Python script to convert image to SVG
│   ├── utils.py            # Utility functions
│   ├── script.sh           # Bash script to process images in batch
│   ├── processed_images_clean # Input directory for images
│   ├── processed_svgs_clean   # Output directory for SVGs
│   ├── __init__.py         # Module initializer
│   └── __pycache__         # Cached Python files
└── tests/                  # Test scripts and configurations
```

## Quick Usage
While in the `scr` folder and installed all dependencies run:
```bash
./script.sh
```
or just without execution permisions
```bash
bash script.sh
```
### 1. Prepare Input Images
Place all image files (PNG, JPG, JPEG) to be processed into the `src/processed_images_clean` directory.

### 2. Build and Run the Docker Container
Build and run the container using Docker Compose:
```bash
docker-compose up --build
```
This method automatically sets up the required environment and processes the images.

### 3. Development with Dev Containers
For development, you can open this project in a Dev Container using Visual Studio Code:
1. Install the "Remote - Containers" extension.
2. Open the project folder in VS Code.
3. Click on "Reopen in Container" when prompted.

### 4. Check Output
Converted SVG files will be saved in the `src/processed_svgs_clean` directory.

## Example
Input:
```
src/processed_images_clean/
├── image1.png
├── image2.jpg
```
Output:
```
src/processed_svgs_clean/
├── image1.svg
├── image2.svg
```

## Error Handling
- If the input directory does not exist, the script will exit with an error message.
- If the Python script encounters an issue, the error will be logged for each affected file.

## Notes
- The Python script (`converter.py`) should handle the conversion logic. Make sure it is configured correctly for the desired output.
- Dependencies are managed using Poetry (`pyproject.toml` and `poetry.lock`).
