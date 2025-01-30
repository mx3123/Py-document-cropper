# README

## Overview

This script automates the process of extracting text from various file formats (images, PDFs, DOCX) using Optical Character Recognition (OCR) powered by Azure Cognitive Services. The script supports image preprocessing, text extraction, and uploading of the processed files to Google Cloud Storage (GCP). It integrates with MongoDB for storing metadata of the processed documents and includes the capability to manage files, including zipping and cleaning up after processing.

## Features

- **Text Extraction**: The script extracts text from:
  - Images (.jpg, .jpeg, .png, .bmp, .tiff) using Azure's Computer Vision API.
  - PDFs by converting each page to an image and applying OCR.
  - DOCX files directly by reading paragraphs.
  
- **File Preprocessing**:
  - Image preprocessing (grayscale, bilateral filtering, adaptive thresholding).
  - Cropping the largest rectangular region in an image for better text extraction.
  - Generation of binary and edge-detected images.
  
- **Storage**:
  - Uploads processed files to Google Cloud Storage.
  - Saves metadata (including extracted text) in MongoDB.
  - Generates zip files containing the original and processed images.

- **Parallel Processing**: Uses `ThreadPoolExecutor` to process files in parallel for faster performance.

- **Cleanup**: After processing, temporary files are cleaned up from both input and output folders.

## Setup Instructions

1. **Install Dependencies**:
   Ensure that the following Python packages are installed:
   - pip install -r packages.txt

2. **Environment Variables**:
   - Create a `.env` file in the root directory of the project and add the following variables:
   - db=your-mongo-db-name
   - documentcollection="your-mongo-collection-name"
   - client=your-mongo-client-url
   - AZURE_VISION_ENDPOINT=your-azure-vision-endpoint
   - AZURE_VISION_KEY=your-azure-vision-key

3. **GCP Service Account Key**:
   - Download your GCP service account JSON file.
   - Transfer the contents to the .env file
   - add the json format to the GCP_Key.json

4. **Image Folder**:
   - Define the path to the image folder where the files will be processed from (set in the script).
   - The output folder will be automatically created if it doesn’t exist.

For more info on this code or if you need archive service's visit https://redsolutions.vercel.app/

