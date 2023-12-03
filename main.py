import os
import fitz
import pytesseract
import re
import logging
import time
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)


# Compile the pattern only once
pattern = re.compile(r'\b\d{5}/\d{3}/\d{2}/\d{3}/\d{2}\b')

def get_nomor_surat(image):
    # Use pytesseract to perform OCR on the image
    text = pytesseract.image_to_string(image)

    # Search for the pattern in the text
    matches = pattern.findall(text)

    # Return the first match
    return matches[0].replace('/', '-') if matches else None

def rename_files(path):
    # Start the timer
    start_time = time.time()

    files = [f for f in os.listdir(path) if f.endswith(".pdf")]
    total_files = len(files)
    processed_files = 0

    logging.info(f"Total PDF files: {total_files}")

    for filename in files:
        doc = fitz.open(os.path.join(path, filename))
        mat = fitz.Matrix(300 / 72, 300 / 72)
        page = doc[0]  # only need the first page
        pix = page.get_pixmap(matrix=mat)
        # Convert the pixmap to a PIL Image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        width, height = image.size
        top_crop = int(height * 0.12)
        bottom_crop = int(height * 0.155)
        
        cropped_img = image.crop((0, top_crop, width, bottom_crop))

        nomor_surat = get_nomor_surat(cropped_img)

        image.close()
        doc.close()

        if nomor_surat:
            new_filename = os.path.join(path, nomor_surat + ".pdf")
            counter = 1

            # If a file with the new filename already exists, append a counter to the new filename
            while os.path.exists(new_filename):
                new_filename = os.path.join(path, f"{nomor_surat}_{counter}.pdf")
                counter += 1

            os.rename(os.path.join(path, filename), new_filename)
            processed_files += 1
            logging.info(f"Processed files: {processed_files}/{total_files}, renamed: {filename} -> {os.path.basename(new_filename)}")
        else:
            logging.info(f"Processed files: {processed_files}/{total_files}, skipped: {filename}")
            
            if not os.path.exists(os.path.join(os.getcwd(), "debug")):
                os.mkdir(os.path.join(os.getcwd(), "debug"))
            
            # remove the pdf extension
            filename = filename[:-4]
            cropped_img.save(os.path.join(os.getcwd(), "debug", filename + ".png"))
            
        cropped_img.close()
                
    # Calculate the time taken
    end_time = time.time()
    time_taken = end_time - start_time

    logging.info(f"Processing completed in {time_taken} seconds")        
    input("Press Enter to exit...")

# rename all pdf files in /pdf folder relative root
rename_files(os.path.join(os.getcwd(), "pdf"))