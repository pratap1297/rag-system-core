"""
Document Classification Module for RAG System
Automatic document categorization with confidence scoring and content analysis
"""

import asyncio
import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
import hashlib

from core.logging_system import get_logger
from core.exceptions import ProcessingError, ConfigurationError
from core.error_handler import with_error_handling
from core.monitoring import get_performance_monitor


class DocumentCategory(Enum):
    """Primary document categories"""
    SAFETY = "safety"
    MAINTENANCE = "maintenance"
    OPERATIONS = "operations"
    TRAINING = "training"
    COMPLIANCE = "compliance"
    GENERAL = "general"


class ClassificationConfidence(Enum):
    """Classification confidence levels"""
    HIGH = "high"      # > 0.8
    MEDIUM = "medium"  # 0.6 - 0.8
    LOW = "low"        # 0.4 - 0.6
    UNCERTAIN = "uncertain"  # < 0.4


@dataclass
class EntityExtraction:
    """Extracted entities from document"""
    equipment_names: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    personnel_roles: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class DocumentAbstract:
    """Document abstract and summary"""
    summary: str
    main_topics: List[str]
    purpose: str
    key_points: List[str]
    word_count: int
    reading_time_minutes: int


@dataclass
class ClassificationRule:
    """Custom classification rule"""
    rule_id: str
    name: str
    description: str
    category: DocumentCategory
    keywords: List[str]
    patterns: List[str]
    priority: int = 1
    enabled: bool = True
    confidence_boost: float = 0.1


@dataclass
class ClassificationResult:
    """Document classification result"""
    document_id: str
    primary_category: DocumentCategory
    secondary_categories: List[DocumentCategory]
    confidence_scores: Dict[DocumentCategory, float]
    overall_confidence: ClassificationConfidence
    
    # Content analysis
    abstract: DocumentAbstract
    entities: EntityExtraction
    
    # Classification metadata
    classification_method: str  # "llama", "rules", "hybrid"
    processing_time: float
    timestamp: datetime
    
    # Quality metrics
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    
    # Rule matching
    matched_rules: List[str] = field(default_factory=list)


@dataclass
class ClassificationFeedback:
    """Feedback for classification improvement"""
    document_id: str
    original_classification: DocumentCategory
    corrected_classification: DocumentCategory
    feedback_type: str  # "correction", "confirmation", "enhancement"
    user_id: str
    timestamp: datetime
    notes: Optional[str] = None


class LlamaMaverickClassifier:
    """LLAMA 4 Maverick integration for document classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("llama_classifier")
        self.api_endpoint = config.get("llama_endpoint", "http://localhost:8080/v1/classify")
        self.model_name = config.get("model_name", "llama-4-maverick")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
    async def classify_document(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document using LLAMA 4 Maverick"""
        
        prompt = self._build_classification_prompt(text, metadata)
        
        try:
            # Simulate LLAMA 4 Maverick API call
            # In production, this would be actual API integration
            result = await self._call_llama_api(prompt)
            return self._parse_classification_response(result)
            
        except Exception as e:
            self.logger.error(f"LLAMA classification failed: {e}")
            raise ProcessingError(f"Classification service error: {e}")
    
    def _build_classification_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Build classification prompt for LLAMA"""
        
        categories = [cat.value for cat in DocumentCategory]
        
        prompt = f"""
Analyze the following document and provide a comprehensive classification.

Document Metadata:
- Title: {metadata.get('title', 'Unknown')}
- File Type: {metadata.get('file_type', 'Unknown')}
- Page Count: {metadata.get('page_count', 'Unknown')}

Document Text (first 2000 characters):
{text[:2000]}

Please provide:
1. Primary category from: {', '.join(categories)}
2. Secondary categories (if applicable)
3. Confidence scores (0.0-1.0) for each category
4. Document abstract (2-3 sentences)
5. Main topics (3-5 key topics)
6. Purpose statement
7. Key entities (equipment, procedures, personnel, locations)

Format your response as JSON with the following structure:
{{
    "primary_category": "category_name",
    "secondary_categories": ["category1", "category2"],
    "confidence_scores": {{"category": 0.85}},
    "abstract": "Document summary...",
    "main_topics": ["topic1", "topic2"],
    "purpose": "Document purpose...",
    "entities": {{
        "equipment": ["item1", "item2"],
        "procedures": ["proc1", "proc2"],
        "personnel": ["role1", "role2"],
        "locations": ["loc1", "loc2"]
    }}
}}
"""
        return prompt
    
    async def _call_llama_api(self, prompt: str) -> Dict[str, Any]:
        """Call LLAMA 4 Maverick API"""
        
        # Placeholder implementation - would be actual API call in production
        await asyncio.sleep(0.1)  # Simulate API latency
        
        # Mock response based on content analysis
        mock_response = {
            "primary_category": "operations",
            "secondary_categories": ["safety"],
            "confidence_scores": {
                "operations": 0.85,
                "safety": 0.65,
                "maintenance": 0.25
            },
            "abstract": "This document provides operational procedures and safety guidelines for equipment maintenance.",
            "main_topics": ["equipment operation", "safety procedures", "maintenance guidelines"],
            "purpose": "To provide comprehensive operational and safety guidance for equipment handling.",
            "entities": {
                "equipment": ["pump", "valve", "control panel"],
                "procedures": ["startup procedure", "shutdown sequence"],
                "personnel": ["operator", "supervisor"],
                "locations": ["control room", "equipment area"]
            }
        }
        
        return mock_response
    
    def _parse_classification_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLAMA classification response"""
        
        try:
            # Validate and normalize response
            primary_category = DocumentCategory(response.get("primary_category", "general"))
            
            secondary_categories = []
            for cat in response.get("secondary_categories", []):
                try:
                    secondary_categories.append(DocumentCategory(cat))
                except ValueError:
                    self.logger.warning(f"Invalid secondary category: {cat}")
            
            confidence_scores = {}
            for cat_name, score in response.get("confidence_scores", {}).items():
                try:
                    category = DocumentCategory(cat_name)
                    confidence_scores[category] = max(0.0, min(1.0, float(score)))
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid confidence score for {cat_name}: {score}")
            
            return {
                "primary_category": primary_category,
                "secondary_categories": secondary_categories,
                "confidence_scores": confidence_scores,
                "abstract": response.get("abstract", ""),
                "main_topics": response.get("main_topics", []),
                "purpose": response.get("purpose", ""),
                "entities": response.get("entities", {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLAMA response: {e}")
            raise ProcessingError(f"Invalid classification response: {e}")


class RuleBasedClassifier:
    """Rule-based document classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("rule_classifier")
        self.rules: List[ClassificationRule] = []
        self.load_default_rules()
    
    def load_default_rules(self):
        """Load default classification rules"""
        
        default_rules = [
            ClassificationRule(
                rule_id="safety_001",
                name="Safety Documentation",
                description="Documents containing safety procedures and guidelines",
                category=DocumentCategory.SAFETY,
                keywords=["safety", "hazard", "risk", "emergency", "accident", "incident", "ppe", "lockout", "tagout"],
                patterns=[r"\bsafety\s+procedure\b", r"\bemergency\s+response\b", r"\brisk\s+assessment\b"],
                priority=2,
                confidence_boost=0.15
            ),
            ClassificationRule(
                rule_id="maintenance_001",
                name="Maintenance Procedures",
                description="Equipment maintenance and repair documentation",
                category=DocumentCategory.MAINTENANCE,
                keywords=["maintenance", "repair", "service", "inspection", "calibration", "preventive", "corrective"],
                patterns=[r"\bmaintenance\s+schedule\b", r"\bpreventive\s+maintenance\b", r"\bwork\s+order\b"],
                priority=2,
                confidence_boost=0.12
            ),
            ClassificationRule(
                rule_id="operations_001",
                name="Operational Procedures",
                description="Standard operating procedures and guidelines",
                category=DocumentCategory.OPERATIONS,
                keywords=["operation", "procedure", "process", "workflow", "standard", "sop", "protocol"],
                patterns=[r"\bstandard\s+operating\s+procedure\b", r"\bsop\b", r"\boperating\s+manual\b"],
                priority=2,
                confidence_boost=0.12
            ),
            ClassificationRule(
                rule_id="training_001",
                name="Training Materials",
                description="Training documentation and educational materials",
                category=DocumentCategory.TRAINING,
                keywords=["training", "education", "course", "curriculum", "learning", "instruction", "tutorial"],
                patterns=[r"\btraining\s+manual\b", r"\bcourse\s+material\b", r"\blearning\s+objective\b"],
                priority=2,
                confidence_boost=0.12
            ),
            ClassificationRule(
                rule_id="compliance_001",
                name="Compliance Documentation",
                description="Regulatory compliance and audit documentation",
                category=DocumentCategory.COMPLIANCE,
                keywords=["compliance", "regulation", "audit", "standard", "certification", "iso", "osha", "epa"],
                patterns=[r"\bcompliance\s+report\b", r"\baudit\s+finding\b", r"\bregulatory\s+requirement\b"],
                priority=2,
                confidence_boost=0.15
            )
        ]
        
        self.rules.extend(default_rules)
        self.logger.info(f"Loaded {len(default_rules)} default classification rules")
    
    def add_rule(self, rule: ClassificationRule):
        """Add custom classification rule"""
        self.rules.append(rule)
        self.logger.info(f"Added classification rule: {rule.name}")
    
    def classify_document(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document using rules"""
        
        text_lower = text.lower()
        category_scores = {cat: 0.0 for cat in DocumentCategory}
        matched_rules = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in rule.keywords if keyword.lower() in text_lower)
            if keyword_matches > 0:
                score += (keyword_matches / len(rule.keywords)) * 0.5
            
            # Check patterns
            pattern_matches = 0
            for pattern in rule.patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_matches += 1
            
            if pattern_matches > 0:
                score += (pattern_matches / len(rule.patterns)) * 0.3
                
            # Apply priority and confidence boost
            if score > 0:
                score = score * rule.priority + rule.confidence_boost
                category_scores[rule.category] += score
                matched_rules.append(rule.rule_id)
        
        # Normalize scores
        max_score = max(category_scores.values()) if any(category_scores.values()) else 1.0
        if max_score > 0:
            category_scores = {cat: score / max_score for cat, score in category_scores.items()}
        
        # Determine primary category
        primary_category = max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Determine secondary categories (score > 0.3)
        secondary_categories = [
            cat for cat, score in category_scores.items() 
            if cat != primary_category and score > 0.3
        ]
        
        return {
            "primary_category": primary_category,
            "secondary_categories": secondary_categories,
            "confidence_scores": category_scores,
            "matched_rules": matched_rules
        }


class MLClassifier:
    """Machine Learning based document classifier"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("ml_classifier")
        # Placeholder for ML model initialization
        
    def classify_document(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document using ML model"""
        # Placeholder implementation
        return {
            "primary_category": "general",
            "confidence_scores": {"general": 0.7},
            "method": "ml"
        }


class DocumentClassifier:
    """Main document classification system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("document_classifier")
        self.monitor = get_performance_monitor()
        
        # Initialize classifiers
        self.llama_classifier = LlamaMaverickClassifier(
            self.config.get("llama_config", {})
        )
        self.rule_classifier = RuleBasedClassifier(
            self.config.get("rules_config", {})
        )
        
        # Configuration
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.use_hybrid_approach = self.config.get("use_hybrid", True)
        self.enable_caching = self.config.get("enable_caching", True)
        
        # Cache for classification results
        self.classification_cache: Dict[str, ClassificationResult] = {}
        
        # Feedback storage
        self.feedback_history: List[ClassificationFeedback] = []
        
        self.logger.info("Document classifier initialized")
    
    @property
    def rules(self) -> List[ClassificationRule]:
        """Get classification rules from rule classifier"""
        return self.rule_classifier.rules
    
    @with_error_handling("document_classifier", "classify_document")
    async def classify_document(
        self, 
        text: str, 
        metadata: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> ClassificationResult:
        """Classify a document"""
        
        start_time = datetime.now()
        
        if not document_id:
            document_id = self._generate_document_id(text, metadata)
        
        # Check cache
        if self.enable_caching and document_id in self.classification_cache:
            self.logger.debug(f"Returning cached classification for {document_id}")
            return self.classification_cache[document_id]
        
        self.logger.info(f"Classifying document {document_id}")
        
        try:
            # Get rule-based classification
            rule_result = self.rule_classifier.classify_document(text, metadata)
            
            # Get LLAMA classification if enabled
            llama_result = None
            if self.use_hybrid_approach:
                try:
                    llama_result = await self.llama_classifier.classify_document(text, metadata)
                except Exception as e:
                    self.logger.warning(f"LLAMA classification failed, using rules only: {e}")
            
            # Combine results
            final_result = self._combine_classification_results(
                rule_result, llama_result, text, metadata
            )
            
            # Create classification result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            classification_result = ClassificationResult(
                document_id=document_id,
                primary_category=final_result["primary_category"],
                secondary_categories=final_result["secondary_categories"],
                confidence_scores=final_result["confidence_scores"],
                overall_confidence=self._determine_confidence_level(final_result["confidence_scores"]),
                abstract=final_result.get("abstract", self._generate_basic_abstract(text)),
                entities=self._extract_entities(text, final_result.get("entities", {})),
                classification_method=final_result["method"],
                processing_time=processing_time,
                timestamp=datetime.now(),
                matched_rules=final_result.get("matched_rules", [])
            )
            
            # Determine if review is needed
            classification_result.needs_review, classification_result.review_reasons = \
                self._assess_review_need(classification_result)
            
            # Cache result
            if self.enable_caching:
                self.classification_cache[document_id] = classification_result
            
            self.logger.info(
                f"Document {document_id} classified as {classification_result.primary_category.value} "
                f"with {classification_result.overall_confidence.value} confidence"
            )
            
            return classification_result
            
        except Exception as e:
            self.logger.error(f"Classification failed for document {document_id}: {e}")
            raise ProcessingError(f"Document classification failed: {e}")
    
    async def classify_batch(
        self, 
        documents: List[Tuple[str, Dict[str, Any]]], 
        max_concurrent: int = 5
    ) -> List[ClassificationResult]:
        """Classify multiple documents concurrently"""
        
        self.logger.info(f"Starting batch classification of {len(documents)} documents")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def classify_single(text_metadata_pair):
            async with semaphore:
                text, metadata = text_metadata_pair
                return await self.classify_document(text, metadata)
        
        tasks = [classify_single(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch classification failed for document {i}: {result}")
            else:
                successful_results.append(result)
        
        self.logger.info(f"Batch classification completed: {len(successful_results)}/{len(documents)} successful")
        return successful_results
    
    def add_feedback(self, feedback: ClassificationFeedback):
        """Add classification feedback for learning"""
        
        self.feedback_history.append(feedback)
        self.logger.info(f"Added classification feedback for document {feedback.document_id}")
        
        # Invalidate cache for this document
        if feedback.document_id in self.classification_cache:
            del self.classification_cache[feedback.document_id]
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        
        total_classifications = len(self.classification_cache)
        
        if total_classifications == 0:
            return {"total_classifications": 0}
        
        # Category distribution
        category_counts = {}
        confidence_distribution = {conf.value: 0 for conf in ClassificationConfidence}
        review_needed = 0
        
        for result in self.classification_cache.values():
            category = result.primary_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            confidence_distribution[result.overall_confidence.value] += 1
            if result.needs_review:
                review_needed += 1
        
        return {
            "total_classifications": total_classifications,
            "category_distribution": category_counts,
            "confidence_distribution": confidence_distribution,
            "review_needed": review_needed,
            "review_percentage": (review_needed / total_classifications) * 100,
            "feedback_count": len(self.feedback_history)
        }
    
    def _combine_classification_results(
        self, 
        rule_result: Dict[str, Any], 
        llama_result: Optional[Dict[str, Any]],
        text: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine rule-based and LLAMA classification results"""
        
        if not llama_result:
            # Rules only
            return {
                **rule_result,
                "method": "rules",
                "abstract": self._generate_basic_abstract(text),
                "entities": {}
            }
        
        # Hybrid approach - combine both results
        combined_scores = {}
        
        # Weight the scores (60% LLAMA, 40% rules)
        llama_weight = 0.6
        rules_weight = 0.4
        
        all_categories = set(rule_result["confidence_scores"].keys()) | \
                        set(llama_result["confidence_scores"].keys())
        
        for category in all_categories:
            llama_score = llama_result["confidence_scores"].get(category, 0.0)
            rule_score = rule_result["confidence_scores"].get(category, 0.0)
            combined_scores[category] = (llama_score * llama_weight) + (rule_score * rules_weight)
        
        # Determine primary category
        primary_category = max(combined_scores.items(), key=lambda x: x[1])[0]
        
        # Determine secondary categories
        secondary_categories = [
            cat for cat, score in combined_scores.items() 
            if cat != primary_category and score > 0.3
        ]
        
        return {
            "primary_category": primary_category,
            "secondary_categories": secondary_categories,
            "confidence_scores": combined_scores,
            "method": "hybrid",
            "abstract": llama_result.get("abstract", self._generate_basic_abstract(text)),
            "main_topics": llama_result.get("main_topics", []),
            "purpose": llama_result.get("purpose", ""),
            "entities": llama_result.get("entities", {}),
            "matched_rules": rule_result.get("matched_rules", [])
        }
    
    def _determine_confidence_level(self, confidence_scores: Dict[DocumentCategory, float]) -> ClassificationConfidence:
        """Determine overall confidence level"""
        
        max_confidence = max(confidence_scores.values()) if confidence_scores else 0.0
        
        if max_confidence > 0.8:
            return ClassificationConfidence.HIGH
        elif max_confidence > 0.6:
            return ClassificationConfidence.MEDIUM
        elif max_confidence > 0.4:
            return ClassificationConfidence.LOW
        else:
            return ClassificationConfidence.UNCERTAIN
    
    def _assess_review_need(self, result: ClassificationResult) -> Tuple[bool, List[str]]:
        """Assess if classification needs manual review"""
        
        reasons = []
        
        # Low confidence
        if result.overall_confidence in [ClassificationConfidence.LOW, ClassificationConfidence.UNCERTAIN]:
            reasons.append("Low classification confidence")
        
        # Close scores between categories
        scores = list(result.confidence_scores.values())
        if len(scores) >= 2:
            scores.sort(reverse=True)
            if scores[0] - scores[1] < 0.2:
                reasons.append("Close confidence scores between categories")
        
        # No clear primary category
        primary_score = result.confidence_scores.get(result.primary_category, 0.0)
        if primary_score < self.confidence_threshold:
            reasons.append("Primary category score below threshold")
        
        return len(reasons) > 0, reasons
    
    def _generate_basic_abstract(self, text: str) -> DocumentAbstract:
        """Generate basic document abstract"""
        
        # Simple extractive summary - first few sentences
        sentences = re.split(r'[.!?]+', text)
        summary_sentences = [s.strip() for s in sentences[:3] if s.strip()]
        summary = '. '.join(summary_sentences)
        
        # Basic topic extraction
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        main_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        main_topics = [topic[0] for topic in main_topics]
        
        word_count = len(words)
        reading_time = max(1, word_count // 200)  # Assume 200 words per minute
        
        return DocumentAbstract(
            summary=summary[:500],  # Limit summary length
            main_topics=main_topics,
            purpose="Document analysis and classification",
            key_points=summary_sentences,
            word_count=word_count,
            reading_time_minutes=reading_time
        )
    
    def _extract_entities(self, text: str, llama_entities: Dict[str, List[str]]) -> EntityExtraction:
        """Extract entities from text"""
        
        # Use LLAMA entities if available, otherwise basic extraction
        if llama_entities:
            return EntityExtraction(
                equipment_names=llama_entities.get("equipment", []),
                procedures=llama_entities.get("procedures", []),
                personnel_roles=llama_entities.get("personnel", []),
                locations=llama_entities.get("locations", []),
                dates=llama_entities.get("dates", []),
                keywords=llama_entities.get("keywords", [])
            )
        
        # Basic entity extraction
        equipment_patterns = [
            r'\b(?:pump|valve|motor|sensor|controller|panel|switch|gauge)\w*\b',
            r'\b(?:equipment|device|instrument|tool|machine)\s+\w+\b'
        ]
        
        procedure_patterns = [
            r'\b(?:procedure|process|step|instruction|guideline)\w*\b',
            r'\b(?:start|stop|shutdown|startup|maintenance|inspection)\s+\w+\b'
        ]
        
        role_patterns = [
            r'\b(?:operator|technician|supervisor|manager|engineer|specialist)\w*\b',
            r'\b(?:personnel|staff|worker|employee)\w*\b'
        ]
        
        equipment = self._extract_by_patterns(text, equipment_patterns)
        procedures = self._extract_by_patterns(text, procedure_patterns)
        roles = self._extract_by_patterns(text, role_patterns)
        
        return EntityExtraction(
            equipment_names=equipment,
            procedures=procedures,
            personnel_roles=roles,
            locations=[],
            dates=[],
            keywords=[]
        )
    
    def _extract_by_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Extract entities using regex patterns"""
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(matches)
        
        return list(entities)[:10]  # Limit to top 10
    
    def _generate_document_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate unique document ID"""
        
        content = f"{text[:1000]}{json.dumps(metadata, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest() 