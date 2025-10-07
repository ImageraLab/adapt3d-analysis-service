from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import io
import requests
from skimage import filters, morphology, measure, feature
from scipy import ndimage
from scipy.signal import find_peaks
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Increase PIL image size limit for large microscopy images
Image.MAX_IMAGE_PIXELS = 200000000  # Allow up to 200 megapixels

app = FastAPI(title="Laboratory Image Analysis Service", version="2.0.0")

class AnalysisRequest(BaseModel):
    image_url: str
    metadata: dict

def download_image(url: str, max_retries: int = 3) -> np.ndarray:
    """Download image from URL with retry logic and extended timeout."""
    logger.info(f"Starting download of image from: {url}")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            
            logger.info(f"Download successful, status code: {response.status_code}")
            
            image_content = io.BytesIO(response.content)
            image = Image.open(image_content)
            
            logger.info(f"Image opened successfully. Size: {image.width}x{image.height}, Mode: {image.mode}")
            
            # Resize if image is too large
            max_dimension = 4000
            if image.width > max_dimension or image.height > max_dimension:
                logger.info(f"Image is large, resizing from {image.width}x{image.height}")
                ratio = min(max_dimension / image.width, max_dimension / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized to {new_size[0]}x{new_size[1]}")
            
            if image.mode != 'RGB':
                logger.info(f"Converting from {image.mode} to RGB")
                image = image.convert('RGB')
            
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
                detail=f"Image download timed out after {max_retries} attempts."
            )
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to process image: {str(e)}"
            )
    
    raise HTTPException(status_code=500, detail="Failed to download image")

def analyze_brain_vessels(image: np.ndarray, metadata: dict) -> dict:
    """
    Advanced analysis for brain vessel and artery detection in 500μm sections.
    Detects vessels from meninges to cortex and predicts arterial patterns.
    """
    try:
        logger.info("Performing brain vessel and artery analysis")
        
        # Get section thickness from metadata
        section_thickness = metadata.get('section_thickness_microns', 500)
        staining_method = metadata.get('staining_method', 'Unknown')
        
        # Convert to grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        if image_gray.max() > 1:
            image_gray = image_gray.astype(float) / 255.0
        
        # Enhance vessel contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_enhanced = clahe.apply((image_gray * 255).astype(np.uint8))
        image_enhanced = image_enhanced.astype(float) / 255.0
        
        # Multi-scale vessel detection
        # Use different sigmas to detect vessels of different sizes
        vessels_small = filters.frangi(image_enhanced, sigmas=range(1, 3), black_ridges=False)
        vessels_medium = filters.frangi(image_enhanced, sigmas=range(3, 6), black_ridges=False)
        vessels_large = filters.frangi(image_enhanced, sigmas=range(6, 10), black_ridges=False)
        
        # Combine multi-scale results
        vessels_combined = np.maximum(vessels_small, np.maximum(vessels_medium, vessels_large))
        
        # Threshold to get binary vessel mask
        vessel_threshold = filters.threshold_otsu(vessels_combined)
        vessel_mask = vessels_combined > vessel_threshold
        
        # Clean up the mask
        vessel_mask = morphology.remove_small_objects(vessel_mask, min_size=20)
        vessel_mask = morphology.binary_closing(vessel_mask, morphology.disk(2))
        
        # Skeletonize to get vessel centerlines
        skeleton = morphology.skeletonize(vessel_mask)
        
        # Label connected vessels
        labeled_vessels = measure.label(vessel_mask)
        vessel_regions = measure.regionprops(labeled_vessels, intensity_image=image_enhanced)
        
        # Categorize vessels by size
        arteries = []
        arterioles = []
        capillaries = []
        
        for region in vessel_regions:
            area = region.area
            major_axis = region.major_axis_length
            minor_axis = region.minor_axis_length
            
            # Estimate vessel diameter (approximate)
            estimated_diameter = np.sqrt(4 * area / np.pi)
            
            # Classify based on size and shape
            if estimated_diameter > 30:  # Large vessels (arteries)
                arteries.append({
                    'area': float(area),
                    'diameter': float(estimated_diameter),
                    'length': float(major_axis),
                    'centroid': [float(region.centroid[0]), float(region.centroid[1])]
                })
            elif estimated_diameter > 10:  # Medium vessels (arterioles)
                arterioles.append({
                    'area': float(area),
                    'diameter': float(estimated_diameter),
                    'length': float(major_axis)
                })
            else:  # Small vessels (capillaries)
                capillaries.append({
                    'area': float(area),
                    'diameter': float(estimated_diameter)
                })
        
        # Calculate vessel density
        total_vessel_area = np.sum(vessel_mask)
        vessel_density = float(total_vessel_area / vessel_mask.size * 100)
        
        # Analyze vessel orientation (tracking from meninges to cortex)
        # Typically vertical or radial patterns
        angles = []
        for region in vessel_regions:
            if region.major_axis_length > 50:  # Only analyze larger vessels
                orientation = region.orientation
                angles.append(np.degrees(orientation))
        
        predominant_direction = "radial" if len(angles) > 0 else "unknown"
        if len(angles) > 0:
            mean_angle = np.mean(angles)
            if -30 < mean_angle < 30:
                predominant_direction = "vertical (meninges to cortex)"
            elif abs(mean_angle) > 60:
                predominant_direction = "horizontal"
        
        # Calculate branching points (indicators of vascular architecture)
        # Use skeleton to find branch points
        branch_points = []
        padded_skeleton = np.pad(skeleton, 1, mode='constant')
        for i in range(1, padded_skeleton.shape[0] - 1):
            for j in range(1, padded_skeleton.shape[1] - 1):
                if padded_skeleton[i, j]:
                    neighbors = np.sum(padded_skeleton[i-1:i+2, j-1:j+2]) - 1
                    if neighbors > 2:  # Branch point
                        branch_points.append([i-1, j-1])
        
        branch_point_count = len(branch_points)
        
        # Estimate vessels per section
        total_vessels = len(arteries) + len(arterioles) + len(capillaries)
        vessels_per_section = total_vessels
        
        # Generate findings
        findings = []
        findings.append(f"Detected {len(arteries)} arteries, {len(arterioles)} arterioles, and {len(capillaries)} capillaries")
        findings.append(f"Total vessels in section: {vessels_per_section}")
        findings.append(f"Vessel density: {vessel_density:.2f}% of tissue area")
        findings.append(f"Identified {branch_point_count} vascular branch points")
        findings.append(f"Predominant vessel direction: {predominant_direction}")
        
        if len(arteries) > 0:
            avg_artery_diameter = np.mean([a['diameter'] for a in arteries])
            findings.append(f"Average artery diameter: {avg_artery_diameter:.1f} pixels")
        
        # Predict arterial patterns
        vascular_patterns = []
        if len(arteries) > 3:
            vascular_patterns.append("Dense arterial network detected")
            vascular_patterns.append("Multiple penetrating arteries identified from meninges")
        elif len(arteries) > 0:
            vascular_patterns.append("Moderate arterial presence")
        
        if branch_point_count > 10:
            vascular_patterns.append("Complex branching pattern suggests rich vascularization")
        
        return {
            "total_vessels": vessels_per_section,
            "arteries_count": len(arteries),
            "arterioles_count": len(arterioles),
            "capillaries_count": len(capillaries),
            "vessel_density_percent": vessel_density,
            "branch_points": branch_point_count,
            "predominant_direction": predominant_direction,
            "section_thickness_microns": section_thickness,
            "staining_method": staining_method,
            "summary": f"Detected {vessels_per_section} total vessels in this {section_thickness}μm brain section, including {len(arteries)} arteries extending from meninges to cortex.",
            "findings": findings,
            "vascular_patterns": vascular_patterns,
            "confidence": min(0.85, 0.5 + (total_vessels / 100)),
            "measurements": {
                "total_vessel_area_pixels": float(total_vessel_area),
                "avg_artery_diameter_pixels": float(np.mean([a['diameter'] for a in arteries])) if arteries else 0,
                "vessel_coverage_percent": vessel_density
            },
            "artery_details": arteries[:10]  # Include details of up to 10 largest arteries
        }
    
    except Exception as e:
        logger.error(f"Brain vessel analysis failed: {str(e)}", exc_info=True)
        raise

def analyze_general_microscopy(image: np.ndarray, metadata: dict) -> dict:
    """Generic analysis for microscopy images."""
    try:
        logger.info("Performing general microscopy analysis")
        
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        if image_gray.max() > 1:
            image_gray = image_gray.astype(float) / 255.0
        
        # Basic image statistics
        mean_intensity = float(np.mean(image_gray))
        std_intensity = float(np.std(image_gray))
        min_intensity = float(np.min(image_gray))
        max_intensity = float(np.max(image_gray))
        
        # Apply thresholding
        blurred = filters.gaussian(image_gray, sigma=2)
        threshold_value = filters.threshold_otsu(blurred)
        binary_image = blurred > threshold_value
        
        # Clean up binary image
        binary_image = morphology.remove_small_objects(binary_image, min_size=10)
        binary_image = morphology.remove_small_holes(binary_image, area_threshold=10)
        
        # Connected component analysis
        labeled_image = measure.label(binary_image)
        regions = measure.regionprops(labeled_image, intensity_image=image_gray)
        
        object_count = len(regions)
        
        findings = []
        findings.append(f"Detected {object_count} distinct objects in the image")
        findings.append(f"Mean intensity: {mean_intensity:.3f}")
        findings.append(f"Intensity range: {min_intensity:.3f} to {max_intensity:.3f}")
        
        if object_count > 0:
            avg_area = np.mean([r.area for r in regions])
            findings.append(f"Average object size: {avg_area:.1f} pixels")
        
        return {
            "object_count": object_count,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "threshold_value": float(threshold_value),
            "summary": f"Analysis detected {object_count} objects with mean intensity of {mean_intensity:.3f}.",
            "findings": findings,
            "confidence": 0.75,
            "measurements": {
                "min_intensity": min_intensity,
                "max_intensity": max_intensity,
                "intensity_std": std_intensity
            }
        }
    
    except Exception as e:
        logger.error(f"Microscopy analysis failed: {str(e)}", exc_info=True)
        raise

def analyze_gel_electrophoresis(image: np.ndarray, metadata: dict) -> dict:
    """Analysis for gel electrophoresis images."""
    try:
        logger.info("Performing gel electrophoresis analysis")
        
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        if image_gray.max() > 1:
            image_gray = image_gray.astype(float) / 255.0
        
        # Invert if bands are dark on light background
        if np.mean(image_gray) > 0.5:
            image_gray = 1 - image_gray
        
        # Detect horizontal bands
        threshold = filters.threshold_otsu(image_gray)
        binary = image_gray > threshold
        
        # Detect lanes (vertical structure)
        vertical_profile = np.sum(binary, axis=0)
        horizontal_profile = np.sum(binary, axis=1)
        
        # Count peaks in horizontal profile as potential bands
        peaks, _ = find_peaks(horizontal_profile, height=np.max(horizontal_profile) * 0.3, distance=10)
        band_count = len(peaks)
        
        # Estimate number of lanes
        lane_peaks, _ = find_peaks(vertical_profile, distance=image_gray.shape[1] // 20)
        lane_count = len(lane_peaks) + 1
        
        findings = []
        findings.append(f"Detected approximately {band_count} bands")
        findings.append(f"Estimated {lane_count} lanes")
        findings.append(f"Mean band intensity: {np.mean(image_gray[binary]):.3f}")
        
        return {
            "band_count": band_count,
            "lane_count": lane_count,
            "mean_intensity": float(np.mean(image_gray)),
            "summary": f"Gel analysis detected {band_count} bands across {lane_count} lanes.",
            "findings": findings,
            "confidence": 0.70,
            "measurements": {
                "mean_band_intensity": float(np.mean(image_gray[binary])),
                "background_intensity": float(np.mean(image_gray[~binary]))
            }
        }
    
    except Exception as e:
        logger.error(f"Gel electrophoresis analysis failed: {str(e)}", exc_info=True)
        raise

def analyze_histology(image: np.ndarray, metadata: dict) -> dict:
    """Analysis for histology/tissue images."""
    try:
        logger.info("Performing histology analysis")
        
        # Analyze color composition
        if len(image.shape) == 3:
            r_channel = image[:, :, 0].astype(float) / 255.0
            g_channel = image[:, :, 1].astype(float) / 255.0
            b_channel = image[:, :, 2].astype(float) / 255.0
            
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
        else:
            image_gray = image.astype(float) / 255.0 if image.max() > 1 else image
            r_channel = g_channel = b_channel = image_gray
        
        # Tissue detection
        threshold = filters.threshold_otsu(image_gray)
        tissue_mask = image_gray < threshold
        
        # Calculate tissue coverage
        tissue_percentage = float(np.sum(tissue_mask) / tissue_mask.size * 100)
        
        # Detect nuclei
        blurred = filters.gaussian(image_gray, sigma=1)
        nuclei_threshold = filters.threshold_otsu(blurred)
        nuclei_mask = blurred < (nuclei_threshold * 0.8)
        
        nuclei_mask = morphology.remove_small_objects(nuclei_mask, min_size=15)
        labeled_nuclei = measure.label(nuclei_mask)
        nuclei_regions = measure.regionprops(labeled_nuclei)
        nuclei_count = len(nuclei_regions)
        
        findings = []
        findings.append(f"Tissue coverage: {tissue_percentage:.1f}%")
        findings.append(f"Detected {nuclei_count} potential nuclei")
        findings.append(f"Mean tissue intensity: {np.mean(image_gray[tissue_mask]):.3f}")
        
        if len(image.shape) == 3:
            findings.append(f"Dominant color channels - R: {np.mean(r_channel):.2f}, G: {np.mean(g_channel):.2f}, B: {np.mean(b_channel):.2f}")
        
        return {
            "tissue_coverage_percent": tissue_percentage,
            "nuclei_count": nuclei_count,
            "mean_intensity": float(np.mean(image_gray)),
            "summary": f"Histology analysis shows {tissue_percentage:.1f}% tissue coverage with {nuclei_count} detected nuclei.",
            "findings": findings,
            "confidence": 0.72,
            "measurements": {
                "tissue_mean_intensity": float(np.mean(image_gray[tissue_mask])),
                "background_mean_intensity": float(np.mean(image_gray[~tissue_mask]))
            }
        }
    
    except Exception as e:
        logger.error(f"Histology analysis failed: {str(e)}", exc_info=True)
        raise

def analyze_fluorescence(image: np.ndarray, metadata: dict) -> dict:
    """Analysis for fluorescence microscopy images."""
    try:
        logger.info("Performing fluorescence analysis")
        
        if len(image.shape) == 3:
            # Analyze individual channels
            channels = []
            for i, color in enumerate(['Red', 'Green', 'Blue']):
                channel = image[:, :, i].astype(float) / 255.0
                channels.append({
                    'name': color,
                    'mean': float(np.mean(channel)),
                    'max': float(np.max(channel)),
                    'std': float(np.std(channel))
                })
            
            image_gray = image[:, :, 1].astype(float) / 255.0
        else:
            image_gray = image.astype(float) / 255.0 if image.max() > 1 else image
            channels = []
        
        # Detect bright fluorescent objects
        threshold = filters.threshold_otsu(image_gray)
        bright_objects = image_gray > threshold
        
        bright_objects = morphology.remove_small_objects(bright_objects, min_size=5)
        
        labeled = measure.label(bright_objects)
        regions = measure.regionprops(labeled, intensity_image=image_gray)
        
        fluorescent_object_count = len(regions)
        
        if fluorescent_object_count > 0:
            avg_intensity = float(np.mean([r.mean_intensity for r in regions]))
            total_fluorescence = float(np.sum(image_gray[bright_objects]))
        else:
            avg_intensity = 0.0
            total_fluorescence = 0.0
        
        findings = []
        findings.append(f"Detected {fluorescent_object_count} fluorescent objects")
        findings.append(f"Average fluorescence intensity: {avg_intensity:.3f}")
        findings.append(f"Total fluorescence signal: {total_fluorescence:.2f}")
        
        result = {
            "fluorescent_object_count": fluorescent_object_count,
            "average_intensity": avg_intensity,
            "total_fluorescence": total_fluorescence,
            "summary": f"Fluorescence analysis detected {fluorescent_object_count} bright objects with average intensity {avg_intensity:.3f}.",
            "findings": findings,
            "confidence": 0.78,
            "measurements": {
                "mean_signal": float(np.mean(image_gray)),
                "max_signal": float(np.max(image_gray))
            }
        }
        
        if channels:
            result["channel_analysis"] = channels
        
        return result
    
    except Exception as e:
        logger.error(f"Fluorescence analysis failed: {str(e)}", exc_info=True)
        raise

def analyze_cell_culture(image: np.ndarray, metadata: dict) -> dict:
    """Analysis for cell culture images."""
    try:
        logger.info("Performing cell culture analysis")
        
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        if image_gray.max() > 1:
            image_gray = image_gray.astype(float) / 255.0
        
        # Detect cells
        blurred = filters.gaussian(image_gray, sigma=2)
        threshold = filters.threshold_otsu(blurred)
        binary = blurred < threshold
        
        binary = morphology.remove_small_objects(binary, min_size=20)
        binary = morphology.remove_small_holes(binary, area_threshold=20)
        
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled, intensity_image=image_gray)
        
        cell_count = len(regions)
        
        # Calculate confluency
        confluency = float(np.sum(binary) / binary.size * 100)
        
        findings = []
        findings.append(f"Detected {cell_count} cells")
        findings.append(f"Estimated confluency: {confluency:.1f}%")
        
        if cell_count > 0:
            avg_cell_area = float(np.mean([r.area for r in regions]))
            findings.append(f"Average cell size: {avg_cell_area:.1f} pixels")
        
        health_score = min(100, confluency * 1.2) if confluency < 80 else 90
        
        return {
            "cell_count": cell_count,
            "confluency_percent": confluency,
            "health_score": health_score,
            "summary": f"Cell culture analysis detected {cell_count} cells with {confluency:.1f}% confluency.",
            "findings": findings,
            "confidence": 0.75,
            "measurements": {
                "mean_intensity": float(np.mean(image_gray)),
                "avg_cell_area": float(np.mean([r.area for r in regions])) if cell_count > 0 else 0
            }
        }
    
    except Exception as e:
        logger.error(f"Cell culture analysis failed: {str(e)}", exc_info=True)
        raise

@app.post("/analyze")
async def analyze_endpoint(request: AnalysisRequest):
    """
    Main endpoint for image analysis.
    Routes to appropriate analysis function based on metadata.
    """
    try:
        logger.info(f"Received analysis request for URL: {request.image_url}")
        logger.info(f"Metadata: {request.metadata}")
        
        # Download the image
        image = download_image(request.image_url)
        
        # Determine analysis type from metadata
        analysis_type = request.metadata.get('analysis_type', 'general')
        
        # Route to appropriate analysis function
        if 'vessel' in analysis_type.lower() or 'brain' in analysis_type.lower() or 'artery' in analysis_type.lower():
            results = analyze_brain_vessels(image, request.metadata)
        elif 'gel' in analysis_type.lower() or 'electrophoresis' in analysis_type.lower():
            results = analyze_gel_electrophoresis(image, request.metadata)
        elif 'histology' in analysis_type.lower() or 'tissue' in analysis_type.lower():
            results = analyze_histology(image, request.metadata)
        elif 'fluorescence' in analysis_type.lower() or 'fluor' in analysis_type.lower():
            results = analyze_fluorescence(image, request.metadata)
        elif 'cell' in analysis_type.lower() and 'culture' in analysis_type.lower():
            results = analyze_cell_culture(image, request.metadata)
        else:
            # Default to general microscopy analysis
            results = analyze_general_microscopy(image, request.metadata)
        
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
        "message": "Laboratory Image Analysis API",
        "version": "2.0",
        "supported_analyses": [
            "Brain Vessel & Artery Analysis (NEW)",
            "General Microscopy",
            "Gel Electrophoresis",
            "Histology/Tissue",
            "Fluorescence Microscopy",
            "Cell Culture"
        ],
        "endpoints": {
            "/analyze": "POST - Analyze laboratory images",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
