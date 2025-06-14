"""
OCR Processing Module for RAG System
Phase 2.2: Advanced OCR with Azure Vision Read 4.0 and LLAMA 4 Maverick fallback
"""

import asyncio
import base64
import io
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import hashlib

# Image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# HTTP client
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Core imports
from core.logging_system import get_logger
from core.monitoring import get_performance_monitor
from core.exceptions import ProcessingError, FileProcessingError
from core.error_handling import with_error_handling


class OCRProvider(Enum):
    """OCR service providers"""
    AZURE_VISION = "azure_vision"
    LLAMA_MAVERICK = "llama_maverick"


class OCRQuality(Enum):
    """OCR quality levels"""
    EXCELLENT = "excellent"  # >95% confidence
    GOOD = "good"           # 85-95% confidence
    FAIR = "fair"           # 70-85% confidence
    POOR = "poor"           # <70% confidence


@dataclass
class TextRegion:
    """Represents a text region with positioning"""
    text: str
    confidence: float
    bounding_box: List[float]  # [x1, y1, x2, y2]
    page_number: int
    region_type: str = "text"  # text, table, header, footer
    language: Optional[str] = None


@dataclass
class OCRPage:
    """OCR results for a single page"""
    page_number: int
    text: str
    confidence: float
    regions: List[TextRegion]
    layout_info: Dict[str, Any]
    processing_time: float
    image_dimensions: Tuple[int, int]
    language: Optional[str] = None


@dataclass
class OCRResult:
    """Complete OCR processing result"""
    text: str
    pages: List[OCRPage]
    overall_confidence: float
    provider: OCRProvider
    fallback_used: bool
    processing_time: float
    quality_metrics: Dict[str, Any]
    layout_analysis: Dict[str, Any]
    language: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ImagePreprocessingResult:
    """Result of image preprocessing"""
    processed_image: bytes
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    enhancements_applied: List[str]
    quality_score: float


class ImagePreprocessor:
    """Image preprocessing for OCR optimization"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("image_preprocessor")
        
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available - image preprocessing disabled")
    
    def preprocess_image(self, image_data: bytes, 
                        enhance_quality: bool = True) -> ImagePreprocessingResult:
        """Preprocess image for optimal OCR"""
        
        if not PIL_AVAILABLE:
            return ImagePreprocessingResult(
                processed_image=image_data,
                original_size=(0, 0),
                processed_size=(0, 0),
                enhancements_applied=[],
                quality_score=0.5
            )
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            original_size = image.size
            enhancements = []
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                enhancements.append("rgb_conversion")
            
            if enhance_quality:
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                enhancements.append("contrast_enhancement")
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
                enhancements.append("sharpness_enhancement")
                
                # Reduce noise
                image = image.filter(ImageFilter.MedianFilter(size=3))
                enhancements.append("noise_reduction")
            
            # Resize if too large (max 4000px on longest side)
            max_size = self.config.get('max_image_size', 4000)
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                enhancements.append("resize")
            
            # Save processed image
            output = io.BytesIO()
            image.save(output, format='PNG', optimize=True)
            processed_data = output.getvalue()
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(image)
            
            return ImagePreprocessingResult(
                processed_image=processed_data,
                original_size=original_size,
                processed_size=image.size,
                enhancements_applied=enhancements,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return ImagePreprocessingResult(
                processed_image=image_data,
                original_size=(0, 0),
                processed_size=(0, 0),
                enhancements_applied=[],
                quality_score=0.3
            )
    
    def _calculate_quality_score(self, image: 'Image.Image') -> float:
        """Calculate image quality score for OCR"""
        
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            
            # Calculate sharpness (Laplacian variance)
            import numpy as np
            img_array = np.array(gray)
            laplacian_var = np.var(np.gradient(img_array))
            
            # Normalize sharpness score
            sharpness_score = min(laplacian_var / 1000, 1.0)
            
            # Calculate contrast
            histogram = gray.histogram()
            contrast_score = (max(histogram) - min(histogram)) / sum(histogram)
            
            # Combined quality score
            quality = (sharpness_score * 0.6 + contrast_score * 0.4)
            return min(max(quality, 0.0), 1.0)
            
        except Exception:
            return 0.5


class AzureVisionOCR:
    """Azure Vision Read 4.0 OCR service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("azure_vision_ocr")
        
        # Azure configuration
        azure_config = config.get('azure', {})
        self.api_key = azure_config.get('computer_vision_key')
        self.endpoint = azure_config.get('computer_vision_endpoint')
        self.api_version = azure_config.get('vision', {}).get('api_version', '2024-02-01')
        
        if not self.api_key or not self.endpoint:
            raise ProcessingError("Azure Vision API key and endpoint required")
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @with_error_handling("azure_vision_ocr", "process_image")
    def process_image(self, image_data: bytes, 
                     language: Optional[str] = None) -> OCRResult:
        """Process image with Azure Vision Read 4.0"""
        
        start_time = time.time()
        
        try:
            # Submit image for processing
            operation_url = self._submit_read_request(image_data, language)
            
            # Poll for results
            result_data = self._poll_read_results(operation_url)
            
            # Parse results
            ocr_result = self._parse_azure_results(result_data, start_time)
            
            self.logger.info(f"Azure Vision OCR completed: {ocr_result.overall_confidence:.3f} confidence")
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"Azure Vision OCR failed: {e}")
            raise ProcessingError(f"Azure Vision OCR processing failed: {str(e)}")
    
    def _submit_read_request(self, image_data: bytes, 
                           language: Optional[str] = None) -> str:
        """Submit image to Azure Vision Read API"""
        
        url = f"{self.endpoint}/vision/v{self.api_version}/read/analyze"
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Content-Type': 'application/octet-stream'
        }
        
        params = {}
        if language:
            params['language'] = language
        
        response = self.session.post(
            url, 
            headers=headers, 
            params=params,
            data=image_data,
            timeout=30
        )
        
        if response.status_code != 202:
            raise ProcessingError(f"Azure Vision API error: {response.status_code} - {response.text}")
        
        operation_url = response.headers.get('Operation-Location')
        if not operation_url:
            raise ProcessingError("No operation URL returned from Azure Vision API")
        
        return operation_url
    
    def _poll_read_results(self, operation_url: str, max_wait: int = 60) -> Dict[str, Any]:
        """Poll Azure Vision Read API for results"""
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key
        }
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = self.session.get(operation_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                raise ProcessingError(f"Azure Vision polling error: {response.status_code}")
            
            result = response.json()
            status = result.get('status', '').lower()
            
            if status == 'succeeded':
                return result
            elif status == 'failed':
                error = result.get('error', {})
                raise ProcessingError(f"Azure Vision processing failed: {error}")
            elif status in ['notstarted', 'running']:
                time.sleep(1)
                continue
            else:
                raise ProcessingError(f"Unknown Azure Vision status: {status}")
        
        raise ProcessingError("Azure Vision processing timeout")
    
    def _parse_azure_results(self, result_data: Dict[str, Any], 
                           start_time: float) -> OCRResult:
        """Parse Azure Vision API results"""
        
        analyze_result = result_data.get('analyzeResult', {})
        read_results = analyze_result.get('readResults', [])
        
        pages = []
        all_text = []
        total_confidence = 0.0
        total_words = 0
        
        for page_data in read_results:
            page_num = page_data.get('page', 1)
            
            # Extract text regions
            regions = []
            page_text = []
            page_confidence = 0.0
            word_count = 0
            
            for line in page_data.get('lines', []):
                line_text = line.get('text', '')
                line_confidence = 1.0  # Azure doesn't provide line confidence
                
                # Get bounding box
                bbox = line.get('boundingBox', [])
                if len(bbox) >= 4:
                    # Convert to [x1, y1, x2, y2] format
                    x_coords = [bbox[i] for i in range(0, len(bbox), 2)]
                    y_coords = [bbox[i] for i in range(1, len(bbox), 2)]
                    bounding_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                else:
                    bounding_box = [0, 0, 0, 0]
                
                region = TextRegion(
                    text=line_text,
                    confidence=line_confidence,
                    bounding_box=bounding_box,
                    page_number=page_num,
                    region_type="text"
                )
                regions.append(region)
                page_text.append(line_text)
                
                # Calculate confidence from words
                words = line.get('words', [])
                for word in words:
                    word_confidence = word.get('confidence', 1.0)
                    page_confidence += word_confidence
                    word_count += 1
            
            # Calculate page metrics
            page_text_str = '\n'.join(page_text)
            page_avg_confidence = page_confidence / max(word_count, 1)
            
            # Get page dimensions
            width = page_data.get('width', 0)
            height = page_data.get('height', 0)
            
            page = OCRPage(
                page_number=page_num,
                text=page_text_str,
                confidence=page_avg_confidence,
                regions=regions,
                layout_info={
                    'width': width,
                    'height': height,
                    'unit': page_data.get('unit', 'pixel'),
                    'angle': page_data.get('angle', 0)
                },
                processing_time=0.0,  # Not available per page
                image_dimensions=(width, height)
            )
            
            pages.append(page)
            all_text.append(page_text_str)
            total_confidence += page_confidence
            total_words += word_count
        
        # Calculate overall metrics
        overall_confidence = total_confidence / max(total_words, 1)
        processing_time = time.time() - start_time
        
        # Determine quality
        quality = self._determine_quality(overall_confidence)
        
        return OCRResult(
            text='\n\n'.join(all_text),
            pages=pages,
            overall_confidence=overall_confidence,
            provider=OCRProvider.AZURE_VISION,
            fallback_used=False,
            processing_time=processing_time,
            quality_metrics={
                'total_words': total_words,
                'average_confidence': overall_confidence,
                'quality_level': quality.value,
                'pages_processed': len(pages)
            },
            layout_analysis={
                'multi_column': False,  # Would need more analysis
                'tables_detected': 0,   # Would need table detection
                'headers_footers': 0    # Would need header/footer detection
            }
        )
    
    def _determine_quality(self, confidence: float) -> OCRQuality:
        """Determine OCR quality level"""
        
        if confidence >= 0.95:
            return OCRQuality.EXCELLENT
        elif confidence >= 0.85:
            return OCRQuality.GOOD
        elif confidence >= 0.70:
            return OCRQuality.FAIR
        else:
            return OCRQuality.POOR


class LlamaMaverickOCR:
    """LLAMA 4 Maverick OCR fallback service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("llama_maverick_ocr")
        
        # This would integrate with LLAMA 4 Maverick OCR service
        # For now, implementing a placeholder structure
        self.logger.info("LLAMA Maverick OCR initialized (placeholder)")
    
    @with_error_handling("llama_maverick_ocr", "process_image")
    def process_image(self, image_data: bytes, 
                     language: Optional[str] = None) -> OCRResult:
        """Process image with LLAMA 4 Maverick OCR"""
        
        start_time = time.time()
        
        # Placeholder implementation
        # In real implementation, this would call LLAMA 4 Maverick OCR API
        
        self.logger.info("Processing with LLAMA Maverick OCR (placeholder)")
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Create placeholder result
        page = OCRPage(
            page_number=1,
            text="[LLAMA Maverick OCR - Placeholder Result]",
            confidence=0.8,
            regions=[],
            layout_info={'width': 800, 'height': 600},
            processing_time=0.5,
            image_dimensions=(800, 600)
        )
        
        return OCRResult(
            text="[LLAMA Maverick OCR - Placeholder Result]",
            pages=[page],
            overall_confidence=0.8,
            provider=OCRProvider.LLAMA_MAVERICK,
            fallback_used=True,
            processing_time=time.time() - start_time,
            quality_metrics={
                'total_words': 5,
                'average_confidence': 0.8,
                'quality_level': OCRQuality.GOOD.value,
                'pages_processed': 1
            },
            layout_analysis={
                'multi_column': False,
                'tables_detected': 0,
                'headers_footers': 0
            },
            warnings=["Using placeholder LLAMA Maverick OCR implementation"]
        )


class OCRProcessor:
    """Main OCR processing orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("ocr_processor")
        self.monitor = get_performance_monitor()
        
        # Initialize components
        self.image_preprocessor = ImagePreprocessor(config)
        
        # Initialize OCR providers
        try:
            self.azure_ocr = AzureVisionOCR(config)
            self.azure_available = True
        except Exception as e:
            self.logger.warning(f"Azure Vision OCR not available: {e}")
            self.azure_available = False
        
        try:
            self.llama_ocr = LlamaMaverickOCR(config)
            self.llama_available = True
        except Exception as e:
            self.logger.warning(f"LLAMA Maverick OCR not available: {e}")
            self.llama_available = False
        
        # Configuration
        self.min_confidence_threshold = config.get('ocr', {}).get('min_confidence', 0.7)
        self.enable_fallback = config.get('ocr', {}).get('enable_fallback', True)
        self.cache_enabled = config.get('ocr', {}).get('cache_enabled', True)
        
        # Simple cache (in production, use Redis or similar)
        self._cache = {}
        
        self.logger.info("OCR Processor initialized")
    
    @with_error_handling("ocr_processor", "process_document")
    def process_document(self, file_path: str, 
                        language: Optional[str] = None,
                        preprocess_images: bool = True) -> OCRResult:
        """Process document with OCR"""
        
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileProcessingError(f"File not found: {file_path}")
            
            self.logger.info(f"Starting OCR processing: {file_path}")
            
            # Check cache
            cache_key = self._generate_cache_key(file_path)
            if self.cache_enabled and cache_key in self._cache:
                self.logger.info("Returning cached OCR result")
                return self._cache[cache_key]
            
            # Read file
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Process based on file type
            if file_path.suffix.lower() in ['.pdf']:
                result = self._process_pdf(file_data, language, preprocess_images)
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                result = self._process_image(file_data, language, preprocess_images)
            else:
                raise ProcessingError(f"Unsupported file type for OCR: {file_path.suffix}")
            
            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = result
            
            self.logger.info(f"OCR processing completed: {result.overall_confidence:.3f} confidence")
            return result
            
        except Exception as e:
            self.logger.error(f"OCR processing failed for {file_path}: {e}")
            raise ProcessingError(f"OCR processing failed: {str(e)}")
    
    def _process_pdf(self, pdf_data: bytes, 
                    language: Optional[str] = None,
                    preprocess_images: bool = True) -> OCRResult:
        """Process PDF document with OCR"""
        
        # For PDF OCR, we would need to:
        # 1. Convert PDF pages to images
        # 2. Process each image with OCR
        # 3. Combine results
        
        # Placeholder implementation - in real system would use pdf2image
        self.logger.warning("PDF OCR processing not fully implemented - using placeholder")
        
        # Create placeholder result
        page = OCRPage(
            page_number=1,
            text="[PDF OCR - Not fully implemented]",
            confidence=0.5,
            regions=[],
            layout_info={'width': 800, 'height': 600},
            processing_time=0.1,
            image_dimensions=(800, 600)
        )
        
        return OCRResult(
            text="[PDF OCR - Not fully implemented]",
            pages=[page],
            overall_confidence=0.5,
            provider=OCRProvider.AZURE_VISION,
            fallback_used=False,
            processing_time=0.1,
            quality_metrics={
                'total_words': 4,
                'average_confidence': 0.5,
                'quality_level': OCRQuality.FAIR.value,
                'pages_processed': 1
            },
            layout_analysis={
                'multi_column': False,
                'tables_detected': 0,
                'headers_footers': 0
            },
            warnings=["PDF OCR processing not fully implemented"]
        )
    
    def _process_image(self, image_data: bytes, 
                      language: Optional[str] = None,
                      preprocess_images: bool = True) -> OCRResult:
        """Process single image with OCR"""
        
        # Preprocess image if enabled
        if preprocess_images:
            preprocessing_result = self.image_preprocessor.preprocess_image(image_data)
            processed_image = preprocessing_result.processed_image
            
            self.logger.info(f"Image preprocessed: {preprocessing_result.enhancements_applied}")
        else:
            processed_image = image_data
        
        # Try primary OCR (Azure Vision)
        if self.azure_available:
            try:
                result = self.azure_ocr.process_image(processed_image, language)
                
                # Check if quality is acceptable
                if result.overall_confidence >= self.min_confidence_threshold:
                    return result
                else:
                    self.logger.warning(f"Azure OCR confidence too low: {result.overall_confidence:.3f}")
                    
                    # Try fallback if enabled
                    if self.enable_fallback and self.llama_available:
                        return self._try_fallback_ocr(processed_image, language, result)
                    else:
                        return result
                        
            except Exception as e:
                self.logger.error(f"Azure OCR failed: {e}")
                
                # Try fallback
                if self.enable_fallback and self.llama_available:
                    return self._try_fallback_ocr(processed_image, language)
                else:
                    raise
        
        # Use fallback OCR if primary not available
        elif self.llama_available:
            return self.llama_ocr.process_image(processed_image, language)
        
        else:
            raise ProcessingError("No OCR providers available")
    
    def _try_fallback_ocr(self, image_data: bytes, 
                         language: Optional[str] = None,
                         primary_result: Optional[OCRResult] = None) -> OCRResult:
        """Try fallback OCR service"""
        
        self.logger.info("Trying fallback OCR (LLAMA Maverick)")
        
        try:
            fallback_result = self.llama_ocr.process_image(image_data, language)
            fallback_result.fallback_used = True
            
            # Compare results if we have both
            if primary_result:
                if fallback_result.overall_confidence > primary_result.overall_confidence:
                    self.logger.info("Fallback OCR produced better results")
                    return fallback_result
                else:
                    self.logger.info("Primary OCR results retained despite low confidence")
                    return primary_result
            else:
                return fallback_result
                
        except Exception as e:
            self.logger.error(f"Fallback OCR also failed: {e}")
            
            # Return primary result if available, otherwise raise
            if primary_result:
                return primary_result
            else:
                raise ProcessingError("Both primary and fallback OCR failed")
    
    def _generate_cache_key(self, file_path: Path) -> str:
        """Generate cache key for file"""
        
        # Use file path and modification time for cache key
        stat = file_path.stat()
        key_data = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_supported_formats(self) -> List[str]:
        """Get supported file formats for OCR"""
        
        return ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate if file can be processed with OCR"""
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, "File does not exist"
            
            if not file_path.is_file():
                return False, "Path is not a file"
            
            # Check file extension
            if file_path.suffix.lower() not in self.get_supported_formats():
                return False, f"Unsupported file format: {file_path.suffix}"
            
            # Check file size
            file_size = file_path.stat().st_size
            max_size = self.config.get('ocr', {}).get('max_file_size', 50 * 1024 * 1024)  # 50MB
            
            if file_size == 0:
                return False, "File is empty"
            
            if file_size > max_size:
                return False, f"File too large: {file_size} bytes (max: {max_size})"
            
            # Check if OCR providers are available
            if not self.azure_available and not self.llama_available:
                return False, "No OCR providers available"
            
            return True, "File is valid for OCR processing"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get OCR processing statistics"""
        
        return {
            'azure_available': self.azure_available,
            'llama_available': self.llama_available,
            'cache_enabled': self.cache_enabled,
            'cached_results': len(self._cache),
            'supported_formats': self.get_supported_formats(),
            'min_confidence_threshold': self.min_confidence_threshold,
            'fallback_enabled': self.enable_fallback
        }
    
    def clear_cache(self):
        """Clear OCR results cache"""
        
        self._cache.clear()
        self.logger.info("OCR cache cleared")
    
    async def process_document_async(self, file_path: str, 
                                   language: Optional[str] = None,
                                   preprocess_images: bool = True) -> OCRResult:
        """Process document with OCR asynchronously"""
        
        # Run synchronous processing in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.process_document, 
            file_path, 
            language, 
            preprocess_images
        )
    
    async def process_batch_async(self, file_paths: List[str],
                                language: Optional[str] = None,
                                max_concurrent: int = 5) -> List[OCRResult]:
        """Process multiple documents with OCR asynchronously"""
        
        self.logger.info(f"Starting async OCR batch processing of {len(file_paths)} files")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(file_path: str) -> OCRResult:
            async with semaphore:
                return await self.process_document_async(file_path, language)
        
        # Process all files concurrently
        tasks = [process_single(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process {file_paths[i]}: {result}")
                
                # Create failed result
                failed_result = OCRResult(
                    text="",
                    pages=[],
                    overall_confidence=0.0,
                    provider=OCRProvider.AZURE_VISION,
                    fallback_used=False,
                    processing_time=0.0,
                    quality_metrics={},
                    layout_analysis={},
                    errors=[str(result)]
                )
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if not r.errors)
        self.logger.info(f"Async OCR batch processing completed: {successful}/{len(file_paths)} successful")
        
        return processed_results 