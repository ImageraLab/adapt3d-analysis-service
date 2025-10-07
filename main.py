from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import io
import requests
from skimage import filters, morphology, measure
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Increase PIL image size limit for large microscopy images
Image.MAX_IMAGE_PIXELS = 200000000  # Allow up to 200 megapixels

app = FastAPI(title="Neuron Analysis Service", version="1.0.0")

class AnalysisRequest(BaseModel):
    image_url: str
    metadata: dict

def download_image(url: str, max_retries: int = 3) -> np.ndarray:
    """Download image from URL with retry logic and extended timeout."""
    logger.info(f"Starting download of image from: {url}")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            
            # Increase timeout to 120 seconds for large images
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            
            logger.info(f"Download successful, status code: {response.status_code}")
            
            # Read the image content
            image_content = io.BytesIO(response.content)
            image = Image.open(image_content)
            
            logger.info(f"Image opened successfully. Size: {image.width}x{image.height}, Mode: {image.mode}")
            
            # Resize if image is too large (to prevent memory issues)
            max_dimension = 4000  # Maximum width or height
            if image.width > max_dimension or image.height > max_dimension:
                logger.info(f"Image is large, resizing from {image.width}x{image.height}")
                ratio = min(max_dimension / image.width, max_dimension / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized to {new_size[0]}x{new_size[1]}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                logger.info(f"Converting from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            logger.info(f"Successfully converted to numpy array: shape {image_array.shape}")
            return image_array
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            raise HTTPException(
                status_code=408, 
                detail=f"Image download timed out after {max_retries} attempts. The image might be too large."
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to download image after {max_retries} attempts: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error processing image: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to process image: {str(e)}"
            )
    
    raise HTTPException(status_code=500, detail="Failed to download image")

def analyze_neuron_image(image: np.ndarray, metadata: dict) -> dict:
    """
    Analyze a neuron microscopy image and extract quantitative metrics.
    """
    try:
        logger.info(f"Starting analysis of image with shape: {image.shape}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        # Normalize to 0-1 range
        if image_gray.max() > 1:
            image_gray = image_gray.astype(float) / 255.0
        
        logger.info("Image preprocessed successfully")
        
        # 1. Preprocessing - Apply Gaussian blur to reduce noise
        blurred = filters.gaussian(image_gray, sigma=2)
        
        # Adaptive thresholding using Otsu's method
        threshold_value = filters.threshold_otsu(blurred)
        binary_image = blurred > threshold_value
        
        logger.info(f"Thresholding complete with value: {threshold_value}")
        
        # 2. Morphological operations - clean up the image
        binary_image = morphology.remove_small_objects(binary_image, min_size=5)
        binary_image = morphology.remove_small_holes(binary_image, area_threshold=5)
        
        # 3. Connected component analysis
        labeled_image = measure.label(binary_image)
        regions = measure.regionprops(labeled_image, intensity_image=image_gray)
        
        logger.info(f"Found {len(regions)} initial regions")
        
        # 4. Filter regions by size and shape (neurons)
        neuron_regions = []
        min_area = 10  # Minimum area for a valid neuron
        max_area = 5000  # Maximum area
        
        for region in regions:
            if min_area < region.area < max_area:
                # Additional shape filtering (optional)
                if region.eccentricity < 0.95:  # Filter out very elongated objects
                    neuron_regions.append(region)
        
        # 5. Calculate metrics
        neuron_count = len(neuron_regions)
        
        logger.info(f"Filtered to {neuron_count} neuron regions")
        
        if neuron_count > 0:
            # Average intensity of detected neurons
            total_intensity = sum([region.mean_intensity for region in neuron_regions])
            avg_intensity = total_intensity / neuron_count
            
            # Calculate signal-to-noise ratio
            foreground = image_gray[binary_image]
            background = image_gray[~binary_image]
            
            if len(background) > 0:
                snr = np.mean(foreground) / np.std(background) if np.std(background) > 0 else 0
            else:
                snr = 0
            
            # Size distribution
            areas = [region.area for region in neuron_regions]
            avg_neuron_size = np.mean(areas)
            
            findings = []
            findings.append(f"Detected {neuron_count} potential neurons")
            findings.append(f"Average neuron size: {avg_neuron_size:.1f} pixels")
            findings.append(f"Average intensity: {avg_intensity:.3f}")
            
            logger.info("Analysis completed successfully")
            
            return {
                "neuron_count": neuron_count,
                "average_intensity": float(avg_intensity),
                "signal_to_noise_ratio": float(snr),
                "summary": f"Analysis detected {neuron_count} neurons with an average fluorescence intensity of {avg_intensity:.2f}.",
                "findings": findings,
                "confidence": min(0.8, neuron_count / 50),
                "measurements": {
                    "avg_neuron_size_pixels": float(avg_neuron_size),
                    "total_neuron_area": float(sum(areas))
                }
            }
        else:
            logger.info("No neurons detected in image")
            return {
                "neuron_count": 0,
                "summary": "No neurons detected in the image. Consider adjusting image quality or threshold parameters.",
                "confidence": 0.0
            }
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze")
async def analyze_endpoint(request: AnalysisRequest):
    """
    Main endpoint for image analysis.
    Accepts an image URL and metadata, returns analysis results.
    """
    try:
        logger.info(f"Received analysis request for URL: {request.image_url}")
        logger.info(f"Metadata: {request.metadata}")
        
        # Download the image
        image = download_image(request.image_url)
        
        # Perform analysis
        results = analyze_neuron_image(image, request.metadata)
        
        logger.info("Returning successful results")
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Neuron Analysis API",
        "version": "1.0",
        "endpoints": {
            "/analyze": "POST - Analyze neuron microscopy images"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
