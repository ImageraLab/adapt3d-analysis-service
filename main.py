import os
import io
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import requests
import openai
from typing import Optional
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Increase PIL image size limit for large microscopy images
Image.MAX_IMAGE_PIXELS = 200000000  # Allow up to 200 megapixels

app = FastAPI()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set!")
else:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key configured successfully")

class AnalysisRequest(BaseModel):
    image_url: str
    metadata: dict = {}

@app.get("/")
async def root():
    return {
        "message": "Laboratory Image Analysis API",
        "version": "2.0",
        "status": "operational",
        "endpoints": {
            "/analyze": "POST - Analyze laboratory images",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "timestamp": time.time()
    }

@app.post("/analyze")
async def analyze_image(request: AnalysisRequest):
    start_time = time.time()
    
    try:
        logger.info(f"Received analysis request for URL: {request.image_url}")
        logger.info(f"Metadata: {request.metadata}")
        
        # Download image with timeout
        logger.info("Starting image download...")
        max_retries = 3
        image_content = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{max_retries}")
                response = requests.get(request.image_url, timeout=90, stream=True)
                response.raise_for_status()
                image_content = response.content
                logger.info(f"Download successful, status code: {response.status_code}, size: {len(image_content)} bytes")
                break
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=408, detail="Image download timed out after 90 seconds")
                logger.warning(f"Download attempt {attempt + 1} timed out, retrying in 2 seconds...")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
                logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}, retrying...")
                time.sleep(2)
        
        if not image_content:
            raise HTTPException(status_code=400, detail="Failed to download image after all retries")
        
        # Open and process image
        try:
            image = Image.open(io.BytesIO(image_content))
            logger.info(f"Image opened successfully. Size: {image.size}, Mode: {image.mode}")
            
            # Resize if image is too large (to prevent memory issues)
            max_dimension = 4000
            if image.width > max_dimension or image.height > max_dimension:
                logger.info(f"Image is large ({image.width}x{image.height}), resizing...")
                ratio = min(max_dimension / image.width, max_dimension / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized to {new_size[0]}x{new_size[1]}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                logger.info(f"Converting from {image.mode} to RGB")
                image = image.convert('RGB')
                
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to open image: {str(e)}")
        
        # Convert to numpy array
        try:
            img_array = np.array(image)
            logger.info(f"Successfully converted to numpy array: shape {img_array.shape}")
        except Exception as e:
            logger.error(f"Failed to convert image to numpy: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
        
        # Determine analysis type
        analysis_type = request.metadata.get('analysis_type', 'general')
        logger.info(f"Analysis type: {analysis_type}")
        
        # Perform analysis based on type
        try:
            if analysis_type == 'brain_vessel_analysis':
                logger.info("Performing brain vessel and artery analysis")
                results = await analyze_brain_vessels(image, img_array, request.metadata)
            elif analysis_type == 'gel_electrophoresis':
                logger.info("Performing gel electrophoresis analysis")
                results = await analyze_gel(image, img_array, request.metadata)
            elif analysis_type == 'histology':
                logger.info("Performing histology analysis")
                results = await analyze_histology(image, img_array, request.metadata)
            elif analysis_type == 'fluorescence':
                logger.info("Performing fluorescence microscopy analysis")
                results = await analyze_fluorescence(image, img_array, request.metadata)
            elif analysis_type == 'cell_culture':
                logger.info("Performing cell culture analysis")
                results = await analyze_cell_culture(image, img_array, request.metadata)
            elif analysis_type == 'microscopy':
                logger.info("Performing microscopy analysis")
                results = await analyze_microscopy(image, img_array, request.metadata)
            else:
                logger.info("Performing general analysis")
                results = await analyze_general(image, img_array, request.metadata)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✓ Analysis completed successfully in {elapsed_time:.2f} seconds")
            
            # Add timing info to results
            results['processing_time_seconds'] = round(elapsed_time, 2)
            
            return results
            
        except openai.APITimeoutError:
            logger.error("OpenAI API request timed out")
            raise HTTPException(status_code=504, detail="AI analysis timed out. The image may be too complex.")
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


async def call_openai_vision(image: Image.Image, prompt: str, system_prompt: str, timeout: int = 120) -> str:
    """Helper function to call OpenAI Vision API with proper error handling"""
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        logger.info("Calling OpenAI Vision API...")
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.3,
            timeout=timeout
        )
        
        result = response.choices[0].message.content
        logger.info(f"OpenAI response received, length: {len(result)} characters")
        return result
        
    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        raise


async def analyze_brain_vessels(image: Image.Image, img_array: np.ndarray, metadata: dict):
    """Analyze brain vessel and artery images"""
    try:
        height, width = img_array.shape[:2]
        
        # Calculate basic statistics
        mean_intensity = float(np.mean(img_array))
        std_intensity = float(np.std(img_array))
        
        prompt = f"""Analyze this brain tissue image for blood vessels and arteries.

Image Information:
- Dimensions: {width}x{height} pixels
- Brain region: {metadata.get('brain_region', 'Not specified')}
- Section thickness: {metadata.get('section_thickness_microns', 'Not specified')} microns
- Staining method: {metadata.get('staining_method', 'Not specified')}
- Mean intensity: {mean_intensity:.2f}

Please provide a detailed analysis including:
1. **Vessel Density**: Estimate the density and distribution of blood vessels
2. **Major Vessels**: Identify and describe any major vessels or arteries visible
3. **Vessel Characteristics**: Note vessel size, morphology, and branching patterns
4. **Pathological Features**: Identify any abnormalities, occlusions, or other pathological features
5. **Quantitative Estimates**: Provide estimates of vessel count, coverage area percentage, or other measurable features
6. **Staining Quality**: Assess the quality and clarity of the staining and imaging
7. **Recommendations**: Suggest any additional analyses or imaging techniques that might be beneficial

Format your response with clear section headers."""

        system_prompt = "You are an expert neuroscientist and vascular biologist specializing in histological analysis of brain tissue. Provide detailed, quantitative analysis when possible."
        
        analysis_text = await call_openai_vision(image, prompt, system_prompt, timeout=180)
        
        # Parse key findings
        findings = []
        if "vessel" in analysis_text.lower():
            findings.append("Blood vessels detected and analyzed")
        if "artery" in analysis_text.lower() or "arteries" in analysis_text.lower():
            findings.append("Arterial structures identified")
        if "density" in analysis_text.lower():
            findings.append("Vascular density assessed")
        
        return {
            "summary": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            "full_analysis": analysis_text,
            "findings": findings,
            "image_dimensions": {"width": width, "height": height},
            "measurements": {
                "mean_intensity": round(mean_intensity, 2),
                "std_intensity": round(std_intensity, 2),
                "image_quality": "good" if std_intensity > 30 else "low_contrast"
            },
            "metadata": metadata,
            "analysis_type": "brain_vessel_analysis",
            "confidence": 0.85
        }
        
    except Exception as e:
        logger.error(f"Error in brain vessel analysis: {str(e)}")
        raise


async def analyze_gel(image: Image.Image, img_array: np.ndarray, metadata: dict):
    """Analyze gel electrophoresis images"""
    try:
        height, width = img_array.shape[:2]
        
        prompt = f"""Analyze this gel electrophoresis image.

Image specifications:
- Dimensions: {width}x{height} pixels
- Gel type: {metadata.get('gel_type', 'Not specified')}
- Sample info: {metadata.get('sample_info', 'Not specified')}

Please provide:
1. **Band Detection**: Identify and count visible bands
2. **Band Characteristics**: Describe band intensity, sharpness, and molecular weight estimates
3. **Lane Analysis**: Analyze each lane separately
4. **Quality Assessment**: Evaluate gel quality, background, smearing, or artifacts
5. **Quantitative Data**: Provide band intensities or relative quantities where possible

Format with clear sections and quantitative data."""

        system_prompt = "You are an expert molecular biologist specializing in gel electrophoresis analysis."
        
        analysis_text = await call_openai_vision(image, prompt, system_prompt)
        
        return {
            "summary": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            "full_analysis": analysis_text,
            "image_dimensions": {"width": width, "height": height},
            "metadata": metadata,
            "analysis_type": "gel_electrophoresis",
            "confidence": 0.80
        }
        
    except Exception as e:
        logger.error(f"Error in gel analysis: {str(e)}")
        raise


async def analyze_histology(image: Image.Image, img_array: np.ndarray, metadata: dict):
    """Analyze histology tissue samples"""
    try:
        height, width = img_array.shape[:2]
        
        prompt = f"""Analyze this histology tissue sample.

Image specifications:
- Dimensions: {width}x{height} pixels
- Tissue type: {metadata.get('tissue_type', 'Not specified')}
- Staining: {metadata.get('staining_method', 'Not specified')}

Please provide:
1. **Tissue Architecture**: Describe overall tissue structure and organization
2. **Cell Types**: Identify different cell types present
3. **Staining Pattern**: Analyze staining distribution and intensity
4. **Pathological Features**: Note any abnormalities or pathological changes
5. **Quantitative Assessment**: Provide cell counts, area measurements, or density estimates

Format with clear medical/scientific sections."""

        system_prompt = "You are an expert pathologist specializing in histological tissue analysis."
        
        analysis_text = await call_openai_vision(image, prompt, system_prompt)
        
        return {
            "summary": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            "full_analysis": analysis_text,
            "image_dimensions": {"width": width, "height": height},
            "metadata": metadata,
            "analysis_type": "histology",
            "confidence": 0.82
        }
        
    except Exception as e:
        logger.error(f"Error in histology analysis: {str(e)}")
        raise


async def analyze_fluorescence(image: Image.Image, img_array: np.ndarray, metadata: dict):
    """Analyze fluorescence microscopy images"""
    try:
        height, width = img_array.shape[:2]
        
        prompt = f"""Analyze this fluorescence microscopy image.

Image specifications:
- Dimensions: {width}x{height} pixels
- Fluorophore: {metadata.get('fluorophore', 'Not specified')}
- Wavelength: {metadata.get('wavelength', 'Not specified')}

Please provide:
1. **Signal Analysis**: Assess fluorescence signal strength and distribution
2. **Cell/Structure Detection**: Identify fluorescently labeled cells or structures
3. **Co-localization**: Note any co-localization patterns (if multiple channels)
4. **Signal Quality**: Evaluate signal-to-noise ratio, photobleaching, artifacts
5. **Quantitative Metrics**: Provide intensity measurements, counts, or area coverage

Format with quantitative data where possible."""

        system_prompt = "You are an expert in fluorescence microscopy and cell biology imaging analysis."
        
        analysis_text = await call_openai_vision(image, prompt, system_prompt)
        
        return {
            "summary": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            "full_analysis": analysis_text,
            "image_dimensions": {"width": width, "height": height},
            "metadata": metadata,
            "analysis_type": "fluorescence",
            "confidence": 0.80
        }
        
    except Exception as e:
        logger.error(f"Error in fluorescence analysis: {str(e)}")
        raise


async def analyze_cell_culture(image: Image.Image, img_array: np.ndarray, metadata: dict):
    """Analyze cell culture images"""
    try:
        height, width = img_array.shape[:2]
        
        prompt = f"""Analyze this cell culture image.

Image specifications:
- Dimensions: {width}x{height} pixels
- Cell type: {metadata.get('cell_type', 'Not specified')}
- Culture condition: {metadata.get('culture_condition', 'Not specified')}

Please provide:
1. **Cell Density**: Estimate confluence percentage and cell density
2. **Cell Morphology**: Describe cell shape, size, and overall health
3. **Growth Pattern**: Note cell distribution and growth characteristics
4. **Cell Count**: Provide estimated cell count if possible
5. **Culture Quality**: Assess contamination, debris, or other issues
6. **Recommendations**: Suggest optimal time for passage or harvest

Format with quantitative metrics."""

        system_prompt = "You are an expert cell biologist specializing in cell culture analysis and microscopy."
        
        analysis_text = await call_openai_vision(image, prompt, system_prompt)
        
        return {
            "summary": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            "full_analysis": analysis_text,
            "image_dimensions": {"width": width, "height": height},
            "metadata": metadata,
            "analysis_type": "cell_culture",
            "confidence": 0.83
        }
        
    except Exception as e:
        logger.error(f"Error in cell culture analysis: {str(e)}")
        raise


async def analyze_microscopy(image: Image.Image, img_array: np.ndarray, metadata: dict):
    """Analyze general microscopy images"""
    try:
        height, width = img_array.shape[:2]
        
        prompt = f"""Analyze this microscopy image.

Image specifications:
- Dimensions: {width}x{height} pixels
- Microscopy type: {metadata.get('microscopy_type', 'Not specified')}
- Magnification: {metadata.get('magnification', 'Not specified')}

Please provide:
1. **Sample Identification**: Identify what the sample appears to be
2. **Key Features**: Describe the most prominent features visible
3. **Quantitative Analysis**: Count cells, measure sizes, or other relevant metrics
4. **Image Quality**: Assess focus, contrast, and overall quality
5. **Scientific Insights**: Provide relevant biological or scientific observations

Format with clear sections and data."""

        system_prompt = "You are an expert microscopist with expertise in various microscopy techniques and biological sample analysis."
        
        analysis_text = await call_openai_vision(image, prompt, system_prompt)
        
        return {
            "summary": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            "full_analysis": analysis_text,
            "image_dimensions": {"width": width, "height": height},
            "metadata": metadata,
            "analysis_type": "microscopy",
            "confidence": 0.78
        }
        
    except Exception as e:
        logger.error(f"Error in microscopy analysis: {str(e)}")
        raise


async def analyze_general(image: Image.Image, img_array: np.ndarray, metadata: dict):
    """General laboratory image analysis"""
    try:
        height, width = img_array.shape[:2]
        
        prompt = f"""Analyze this laboratory or scientific image.

Image specifications:
- Dimensions: {width}x{height} pixels
- Additional metadata: {metadata}

Please provide:
1. **Image Type**: Identify what type of laboratory image this is
2. **Key Observations**: Describe the most important features
3. **Quantitative Analysis**: Provide any measurable data or counts
4. **Quality Assessment**: Evaluate image quality and clarity
5. **Scientific Context**: Explain the relevance and what can be learned
6. **Recommendations**: Suggest further analysis or imaging approaches

Be thorough and scientific in your analysis."""

        system_prompt = "You are an expert in laboratory image analysis with broad expertise in microscopy, molecular biology, and scientific imaging."
        
        analysis_text = await call_openai_vision(image, prompt, system_prompt)
        
        return {
            "summary": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            "full_analysis": analysis_text,
            "image_dimensions": {"width": width, "height": height},
            "metadata": metadata,
            "analysis_type": "general",
            "confidence": 0.75
        }
        
    except Exception as e:
        logger.error(f"Error in general analysis: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
