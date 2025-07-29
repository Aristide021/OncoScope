"""
Ollama Client for OncoScope AI Inference
"""
import aiohttp
import json
import logging
from typing import Dict, List, Optional
import asyncio

from ...backend.config import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for Ollama local AI inference"""
    
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name or settings.ollama_model_name
        self.base_url = base_url or settings.ollama_base_url
        self.timeout = aiohttp.ClientTimeout(total=settings.ollama_timeout)
    
    async def analyze_cancer_mutation(self, gene: str, variant: str) -> Dict:
        """Analyze cancer mutation using fine-tuned Gemma 3n"""
        
        prompt = self._create_mutation_analysis_prompt(gene, variant)
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                response = await self._generate(session, prompt)
                
                if response:
                    return self._parse_ai_response(response, gene, variant)
                else:
                    return self._fallback_analysis(gene, variant)
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout analyzing {gene}:{variant}")
            return self._fallback_analysis(gene, variant)
        except Exception as e:
            logger.error(f"AI analysis failed for {gene}:{variant}: {e}")
            return self._fallback_analysis(gene, variant)
    
    def _create_mutation_analysis_prompt(self, gene: str, variant: str) -> str:
        """Create structured prompt for mutation analysis"""
        return f"""Analyze the cancer mutation {gene}:{variant} and provide structured clinical assessment.

Gene: {gene}
Variant: {variant}

Provide your analysis in the following JSON format:
{{
    "pathogenicity": <float between 0.0-1.0>,
    "cancer_types": ["<list of associated cancer types>"],
    "protein_change": "<predicted protein change notation>",
    "mechanism": "<biological mechanism of the mutation>",
    "significance": "<one of: PATHOGENIC, LIKELY_PATHOGENIC, UNCERTAIN, LIKELY_BENIGN, BENIGN>",
    "therapies": ["<list of potential targeted therapies if any>"],
    "prognosis": "<one of: poor, moderate, good, excellent_with_therapy, uncertain>",
    "clinical_context": "<brief clinical interpretation>",
    "confidence": <float between 0.0-1.0 indicating confidence in this analysis>
}}

Focus on established clinical knowledge and cancer genomics databases. If the mutation is unknown, provide your best assessment based on the gene function and variant type."""
    
    async def _generate(self, session: aiohttp.ClientSession, prompt: str) -> Optional[str]:
        """Generate response from Ollama"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.model_temperature,
                "top_p": settings.model_top_p,
                "num_predict": settings.model_max_tokens,
                "seed": 42  # For reproducibility
            }
        }
        
        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', '')
                else:
                    logger.error(f"Ollama API returned status {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None
    
    def _parse_ai_response(self, ai_text: str, gene: str, variant: str) -> Dict:
        """Parse JSON response from AI"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', ai_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                # Validate and clean the response
                return {
                    "pathogenicity": max(0.0, min(1.0, float(data.get("pathogenicity", 0.5)))),
                    "cancer_types": data.get("cancer_types", ["unknown"]),
                    "protein_change": data.get("protein_change", f"p.{variant}"),
                    "mechanism": data.get("mechanism", "Unknown mechanism"),
                    "significance": data.get("significance", "UNCERTAIN"),
                    "therapies": data.get("therapies", []),
                    "prognosis": data.get("prognosis", "uncertain"),
                    "clinical_context": data.get("clinical_context", "Insufficient data"),
                    "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                }
            else:
                logger.warning(f"No JSON found in AI response for {gene}:{variant}")
                return self._fallback_analysis(gene, variant)
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            return self._fallback_analysis(gene, variant)
    
    def _fallback_analysis(self, gene: str, variant: str) -> Dict:
        """Fallback analysis when AI fails"""
        # Basic heuristics based on gene and variant type
        pathogenicity = 0.5
        significance = "UNCERTAIN"
        
        # Known cancer genes get higher baseline
        cancer_genes = {
            "TP53", "KRAS", "EGFR", "BRCA1", "BRCA2", "PIK3CA", 
            "PTEN", "APC", "MLH1", "MSH2", "VHL", "RB1"
        }
        
        if gene in cancer_genes:
            pathogenicity = 0.7
            significance = "LIKELY_PATHOGENIC"
        
        # Frameshift and nonsense variants are usually bad
        if "fs" in variant or "X" in variant or "del" in variant:
            pathogenicity = 0.8
            significance = "LIKELY_PATHOGENIC"
        
        return {
            "pathogenicity": pathogenicity,
            "cancer_types": ["unknown"],
            "protein_change": f"p.{variant}" if not variant.startswith("p.") else variant,
            "mechanism": "Unknown mechanism - AI analysis unavailable",
            "significance": significance,
            "therapies": [],
            "prognosis": "uncertain",
            "clinical_context": "Manual review recommended - AI analysis failed",
            "confidence": 0.3
        }
    
    async def check_model_availability(self) -> bool:
        """Check if the OncoScope model is available in Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        return any(self.model_name in model.get('name', '') for model in models)
            return False
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
    
    async def pull_model(self) -> bool:
        """Pull the OncoScope model if not available"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"name": self.model_name}
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json=payload
                ) as response:
                    if response.status == 200:
                        # Stream the response to show progress
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line)
                                    if 'status' in data:
                                        logger.info(f"Model pull: {data['status']}")
                                except:
                                    pass
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False