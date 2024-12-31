#!/bin/bash

# Define the source directory containing the images
input_dir="./processed_images_clean"

# Define the output directory for the processed images
output_dir="./processed_svgs_clean"

# Check if input directory exists
if [[ ! -d "$input_dir" ]]; then
    echo "Input directory does not exist: $input_dir"
    exit 1
fi

# Check if output directory exists; if not, create it
if [[ ! -d "$output_dir" ]]; then
    echo "Output directory does not exist. Creating: $output_dir"
    mkdir -p "$output_dir"
fi

# Loop through each image in the input directory
for input_image in "$input_dir"/*; do
    # Check if the file is an image (you can extend this to other formats)
    if [[ $input_image == *.png || $input_image == *.jpg || $input_image == *.jpeg ]]; then
        # Get the filename without the directory path
        filename=$(basename "$input_image")
        
        # Define the output file path in the output directory
        output_image="$output_dir/$filename"

        # Run the Python script for each image
        python3 converter.py "$input_image" "$output_image"
        
        # Check if the Python script was successful
        if [[ $? -eq 0 ]]; then
            echo "Processed $filename and saved to $output_image"
        else
            echo "Error processing $filename"
        fi
    fi
done
