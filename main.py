import os
import logging
import fitz
from PIL import Image
import cv2
import numpy as np
import pytesseract
import re
import time


# Set up logging
logging.basicConfig(level=logging.INFO)

def read_all_pdf(path):
    # start the timer
    start_time = time.time()
    
    files = [f for f in os.listdir(path) if f.endswith(".pdf")]
    total_files = len(files)
    
    logging.info(f"Total PDF files: {total_files}")
    
    processed_files = 0
    success_files = 0
    
    for filename in files:
        img = convert_first_page_pdf_to_image(os.path.join(path, filename))
        img = image_preprocessing(img)
              
        # if get nomor surat is not None, rename the file
        new_filename = get_nomor_surat(img, filename)
        if new_filename:
            success_files += 1
            rename_pdf_file(os.path.join(path, filename + ".pdf"), filename, new_filename + ".pdf")
            logging.info(f"({processed_files + 1}/{total_files}): Processing {filename}, Renamed {filename} -> {new_filename}.pdf")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        else:
            logging.info(f"({processed_files + 1}/{total_files}): Processing {filename}, Skipped")        
            
        processed_files += 1
    
    print()
    
    logging.info(f"Total success files: {success_files}/{total_files} = ({success_files/total_files*100:.2f}%)")
    
    print()
    
    # end the timer
    end_time = time.time()
    # calculate the total time
    total_time = end_time - start_time
    logging.info(f"Total time: {total_time:.2f} seconds")
    
    input("Press enter to exit...")

def convert_first_page_pdf_to_image(pdf_path):
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(300 / 72, 300 / 72)
    page = doc[0]  # only need the first page
    pix = page.get_pixmap(matrix=mat)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    return image

def image_preprocessing(img):
    # Crop the image
    img = crop_image(img)
    
    # Deskew the image
    img = deskew(np.array(img))
    
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_thresh

# create a function to crop the image
def crop_image(img):
    # Convert the image to a numpy array
    img = np.array(img)

    height, width = img.shape[:2]
    top_crop = int(height * 0.075)
    bottom_crop = int(height * 0.175)
    right_crop = int(width * 0.5)

    # Crop the image
    cropped_img = img[top_crop:bottom_crop, 0:right_crop]

    return cropped_img

def deskew(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Use the Hough transform to detect lines in the image
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # Calculate the angles of the lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Calculate the median angle and use it as the skew angle
    angle = np.median(angles)

    # Rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed

# Compile the pattern only once
pattern = re.compile(r'\b\d{5}/\d{3}/\d{2}/\d{3}/\d{2}\b')

def get_nomor_surat(image, filename):
    # Convert the OpenCV image (numpy array) to a PIL image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Erode the image to reduce the boldness of the text
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(np.array(image), kernel, iterations = 1)
    
    # Use pytesseract to perform OCR on the image
    text = pytesseract.image_to_string(image, config='--psm 6 --oem 1 -c dpi=300')
    
    # Search for the pattern in the text
    matches = pattern.findall(text)
    
    # if no matches found, write the text to debug folder
    if not matches:
        with open(os.path.join(os.getcwd(), "debug", filename + ".txt"), "w") as f:
            f.write(text)

    # Return the first match
    return matches[0].replace('/', '-') if matches else None

def rename_pdf_file(pdf_path, old_filename, new_filename):
    # Get the directory of the pdf file
    dir_path = os.path.dirname(pdf_path)

    # Check if a file with the new name already exists
    if os.path.exists(os.path.join(dir_path, new_filename)):
        # If it does, append a number to the new name to make it unique
        base, ext = os.path.splitext(new_filename)
        i = 1
        while os.path.exists(os.path.join(dir_path, f"{base}_{i}{ext}")):
            i += 1
        new_filename = f"{base}_{i}{ext}"

    # Rename the file
    os.rename(os.path.join(dir_path, old_filename), os.path.join(dir_path, new_filename))
    

read_all_pdf(os.path.join(os.getcwd(), "pdf"))