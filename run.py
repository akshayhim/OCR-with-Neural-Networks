import os
import re
import base64
import cv2
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

def preprocess_image(image, save_steps=True, preprocess_dir="./preprocessing_result", filename="image"):
    """Preprocess image to improve OCR quality and save all steps."""
    os.makedirs(preprocess_dir, exist_ok=True)
    
    original_copy = image.copy()
    cv2.imwrite(os.path.join(preprocess_dir, f"{filename}_1_original.png"), original_copy)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(preprocess_dir, f"{filename}_2_grayscale.png"), gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(os.path.join(preprocess_dir, f"{filename}_3_blurred.png"), blurred)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(os.path.join(preprocess_dir, f"{filename}_4_threshold.png"), thresh)
    
    inverted = cv2.bitwise_not(thresh)
    cv2.imwrite(os.path.join(preprocess_dir, f"{filename}_5_final.png"), inverted)
    
    print(f"All preprocessing steps saved to {preprocess_dir}")
    return inverted

def encode_image(image_path, preprocess=False):
    """Encode the image to base64, with optional preprocessing."""
    try:
        if preprocess:
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image from {image_path}")
            
            filename = os.path.splitext(os.path.basename(image_path))[0]

            # Preprocess the image
            preprocessed_image = preprocess_image(
                image,
                save_steps=True,
                preprocess_dir="./preprocessing_result",
                filename=filename
            )

            _, buffer = cv2.imencode('.png', preprocessed_image)  # Use png for lossless encoding
            return base64.b64encode(buffer).decode('utf-8')
        else:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_ocr(image_path, output_dir="./ocr_results", preprocess=False):
    """Process image with Mistral OCR and save results to markdown file."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the MISTRAL_API_KEY environment variable.")
    
    client = Mistral(api_key=api_key)
    
    base64_image = encode_image(image_path, preprocess=preprocess)
    if not base64_image:
        return
    
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpg;base64,{base64_image}"
            },
            include_image_base64=True
        )
    except Exception as e:
        print(f"OCR processing failed: {str(e)}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, page in enumerate(ocr_response.pages):
        md = page.markdown
        
        if hasattr(page, 'images') and page.images:
            for image in page.images:
                if hasattr(image, 'image_base64') and image.image_base64:
                    pattern = rf'\(({image.id})\)'
                    md = re.sub(pattern, f'({image.image_base64})', md)
        
        output_file = os.path.join(output_dir, f"ocr_output_page_{i+1}.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md)
        
        print(f"OCR results for page {i+1} saved to {output_file}")
    
    return ocr_response

if __name__ == "__main__":
    image_path = "./test_docs/t2.jpg"
    
    # Set preprocess=True if you want to apply image preprocessing
    process_ocr(image_path, preprocess=True)
