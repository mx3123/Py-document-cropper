import cv2
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from imutils.perspective import four_point_transform
from pymongo import MongoClient
import numpy as np
import os
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from dotenv import load_dotenv
import fitz
import docx
import io
from PIL import Image
import zipfile
from google.cloud import storage
from google.oauth2 import service_account
import datetime
import shutil

load_dotenv()

# MongoDB setup
DATABASE_NAME = os.getenv("db")
COLLECTION_NAME = os.getenv("documentcollection")

mongouri = os.getenv("client")
client = MongoClient(mongouri)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

IMAGE_FOLDER = Path(r"C:\path\to\storage")
PROCESSED_FOLDER = Path(r"C:\path\to\output")
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# this loads the GCP service account info from .env

# Define the path where the GCP_key.json will be saved
SERVICE_KEY_PATH = Path(r"C:\path\to\GCP_key.json")

# List of environment variables corresponding to the service account .env fields
service_account_fields = [
    "type",
    "project_id",
    "private_key_id",
    "private_key",
    "client_email",
    "client_id",
    "auth_uri",
    "token_uri",
    "auth_provider_x509_cert_url",
    "client_x509_cert_url",
    "universe_domain",
]

# Populate the GCP_key.json using environment variables
service_account_data = {field: os.getenv(field.upper()) for field in service_account_fields}

# Ensure all required environment variables are provided
missing_fields = [field for field, value in service_account_data.items() if not value]
if missing_fields:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")

# Write the JSON data to the GCP_key.json file
SERVICE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
with SERVICE_KEY_PATH.open("w") as f:
    json.dump(service_account_data, f, indent=4)

# Load the service account credentials
credentials = service_account.Credentials.from_service_account_file(SERVICE_KEY_PATH)


# Azure Computer Vision setup
VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
VISION_KEY = os.getenv("AZURE_VISION_KEY")

# Initialize Google Cloud Storage client with the explicit credentials, add your bucket!
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.bucket("UR_BUCKET_HERE")

    
def extract_text_with_azure(image_path):
    #Extract text from image using Azure Computer Vision
    try:
        # Create client
        client = ImageAnalysisClient(
            endpoint=VISION_ENDPOINT,
            credential=AzureKeyCredential(VISION_KEY)
        )
        
        # Open image file
        with open(image_path, "rb") as image_file:
            # Analyze image
            result = client.analyze(
                image_data=image_file,
                visual_features=[VisualFeatures.READ]
            )
        
        # Extract text from result
        if result.read is not None:
            text_lines = []
            for block in result.read.blocks:
                for line in block.lines:
                    text_lines.append(line.text)
            return '\n'.join(text_lines)
        
        return ""
    
    except Exception as e:
        print(f"Azure OCR Error: {e}")
        return ""

def extract_text_from_pdf_with_azure(pdf_path):
    #Extract text from PDF using Azure OCR
    try:
        # Convert PDF pages to images
        doc = fitz.open(pdf_path)
        extracted_text = []
        
        for page in doc:
            # Convert page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Save temporary image for Azure OCR
            temp_image_path = PROCESSED_FOLDER / f"temp_page_{uuid.uuid4()}.jpg"
            img.save(temp_image_path)
            
            # Extract text using Azure
            page_text = extract_text_with_azure(str(temp_image_path))
            extracted_text.append(page_text)
            
            # Remove temporary image
            os.unlink(temp_image_path)
        
        doc.close()
        return '\n'.join(extracted_text)
    
    except Exception as e:
        print(f"Azure PDF processing error: {e}")
        return ""

def extract_text_from_docx(docx_path):
    #Extract text from DOCX file.
    try:
        doc = docx.Document(docx_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Error processing DOCX: {e}")
        return ""

def zip_processed_files(file_path):
    #Create a zip archive containing the original file.
    file_path = Path(file_path)
    base_name = file_path.stem
    zip_path = PROCESSED_FOLDER / f"{base_name}.zip"
    
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add the original file
            zipf.write(file_path, file_path.name)
            
            # Add any processed files if they exist
            for processed_file in PROCESSED_FOLDER.iterdir():
                if processed_file.is_file() and processed_file.name.startswith(('cropped_', 'binary_', 'edges_')):
                    zipf.write(processed_file, processed_file.name)
            
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
            print(f"Successfully created zip file: {zip_path}")
            return zip_path
        else:
            print("Warning: Created zip file is empty")
            return None
            
    except Exception as e:
        print(f"Error creating zip file: {e}")
        return None

def upload_to_gcp(file_path):
    #Upload file to GCP and generate signed URL.
    blob_name = f"{file_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(file_path))  # Upload the file directly to the bucket
    
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(days=7),
        method="GET"
    )
    return url

def process_image(file_path):
    #Process files of various types with fallback for unsupported/unprocessable files.
    print(f"Processing: {file_path}")
    file_path = Path(file_path)
    extracted_text = ""
    cropped_path = binary_path = edges_path = None

    try:
        supported_image_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        file_type = file_path.suffix.lower()

        # Try to process the file based on type
        try:
            if file_type in supported_image_formats:
                image = cv2.imread(str(file_path))
                if image is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Enhanced preprocessing
                    # Add bilateral filtering to reduce noise while preserving edges
                    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
                    
                    # Adaptive thresholding instead of Otsu's method
                    binary = cv2.adaptiveThreshold(denoised, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
                    
                    edges = cv2.Canny(denoised, 30, 200) 
                    
                    # Dilates edges to connect fragmented contours
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    edges = cv2.dilate(edges, kernel, iterations=2)
                    
                    # Save intermediate results
                    base_name = file_path.stem
                    binary_path = PROCESSED_FOLDER / f"binary_{base_name}.jpg"
                    edges_path = PROCESSED_FOLDER / f"edges_{base_name}.jpg"
                    cv2.imwrite(str(binary_path), binary)
                    cv2.imwrite(str(edges_path), edges)
                    
                    # Find contours with different approach
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Sort contours by area, filter out very small contours
                    contours = [c for c in contours if cv2.contourArea(c) > 100]  # Adjust area threshold as needed
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                # Find largest rectangular contour
                for c in contours:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:  # Quadrilateral detected
                        warped = four_point_transform(image, approx.reshape(4, 2))
                        unique_id = str(uuid.uuid4())
                        cropped_path = PROCESSED_FOLDER / f"cropped_{unique_id}.jpg"
                        cv2.imwrite(str(cropped_path), warped)
                        
                        # Extect with Azure OCR
                        extracted_text = extract_text_with_azure(str(cropped_path))
                        break
                    
                if cropped_path:
                    extracted_text = extract_text_with_azure(str(cropped_path))
                
                if not extracted_text:
                    # Fallback to full image OCR if crop fails
                    extracted_text = extract_text_with_azure(str(file_path))
            
            elif file_type == '.pdf':
                extracted_text = extract_text_from_pdf_with_azure(str(file_path))
            
            elif file_type == '.docx':
                extracted_text = extract_text_from_docx(str(file_path))
            
        except Exception as e:
            print(f"Error processing file contents: {e}")

        # Create zip file
        zip_path = zip_processed_files(file_path)
        if not zip_path:
            print("Failed to create zip file, uploading original file")
            zip_path = file_path

        # Upload to GCP directly to the bucket
        gcp_url = upload_to_gcp(zip_path)

        # Save to MongoDB
        record = {
            "original_file_path": str(file_path),
            "file_type": file_type,
            "cropped_image_path": gcp_url,
            "file_name": file_path.name,
            "extracted_text": extracted_text,
            "processing_status": "fully_processed" if extracted_text else "uploaded_only"
        }
        
        result = collection.insert_one(record)
        print(f"Document processed and saved with ID: {result.inserted_id}")

        # Cleanup processed files, this clears storage and output
        cleanup_processed_files()

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise


def cleanup_processed_files():
    #Clean up processed files after successful upload.
    try:
        for file_path in PROCESSED_FOLDER.iterdir():
            if file_path.is_file():
                print(f"Cleaning up processed file: {file_path}")
                file_path.unlink()
    except Exception as e:
        print(f"Error during cleanup of processed files: {e}")

def process_files_in_folder(folder_path):
    #Process all files in a folder.
    folder_path = Path(folder_path)
    try:
        with ThreadPoolExecutor() as executor:
            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    executor.submit(process_image, str(file_path))
        
        cleanup_input_files()
    except Exception as e:
        print(f"Error in folder processing: {e}")

def cleanup_input_files():
   #Clean up storage files after processing.
    try:
        for file_path in IMAGE_FOLDER.iterdir():
            if file_path.is_file():
                print(f"Cleaning up input file: {file_path}")
                file_path.unlink()
    except Exception as e:
        print(f"Error during cleanup of input files: {e}")

if __name__ == "__main__":
    try:
        process_files_in_folder(IMAGE_FOLDER)
    except Exception as e:
        print(f"Error in main process: {e}")