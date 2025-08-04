"""
Ollama Client for OncoScope AI Inference
"""
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Tuple
import asyncio

from backend.config import settings

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
        logger.info(f"Analyzing {gene}:{variant} with Gemma 3n model")
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                response = await self._generate(session, prompt)
                
                if response:
                    logger.info(f"Got AI response for {gene}:{variant}, length: {len(response)}")
                    parsed = self._parse_ai_response(response, gene, variant)
                    logger.info(f"Parsed AI response: {parsed}")
                    return parsed
                else:
                    return self._fallback_analysis(gene, variant)
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout analyzing {gene}:{variant}")
            return self._fallback_analysis(gene, variant)
        except Exception as e:
            logger.error(f"AI analysis failed for {gene}:{variant}: {e}")
            return self._fallback_analysis(gene, variant)
    
    async def analyze_multi_mutations(self, prompt: str) -> Dict:
        """Analyze multiple mutations with clustering context using fine-tuned Gemma 3n"""
        logger.info("Gemma 3n analyzing multiple mutations...")
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                response = await self._generate(session, prompt)
                
                if response:
                    logger.info(f"Gemma 3n raw response (first 500 chars): {response[:500]}")
                    logger.info(f"Full response length: {len(response)} chars")
                    # Log last 200 chars to see where it cuts off
                    logger.info(f"Response end (last 200 chars): {response[-200:]}")
                    parsed = self._parse_multi_mutation_response(response)
                    
                    # If we didn't get individual analyses, extract mutations and analyze them
                    if not parsed.get("individual_analyses"):
                        logger.info("Extracting mutations from prompt for individual analysis...")
                        mutations = self._extract_mutations_from_prompt(prompt)
                        parsed["individual_analyses"] = await self._analyze_individual_mutations(mutations)
                    
                    return parsed
                else:
                    return self._fallback_multi_mutation_analysis()
                    
        except asyncio.TimeoutError:
            logger.error("Timeout analyzing multiple mutations")
            return self._fallback_multi_mutation_analysis()
        except Exception as e:
            logger.error(f"AI multi-mutation analysis failed: {e}")
            return self._fallback_multi_mutation_analysis()
    
    def _create_mutation_analysis_prompt(self, gene: str, variant: str) -> str:
        """Create structured prompt for mutation analysis"""
        
        # Add specific guidance for well-known mutations
        context_hint = ""
        if gene == "BRAF" and variant == "c.1799T>A":
            context_hint = "\nNote: c.1799T>A in BRAF codes for the V600E mutation, one of the most important targetable mutations in cancer."
        elif gene == "KRAS" and variant == "c.35G>A":
            context_hint = "\nNote: c.35G>A in KRAS codes for the G12D mutation."
        elif gene == "EGFR" and variant == "c.2369C>T":
            context_hint = "\nNote: c.2369C>T in EGFR codes for the T790M mutation."
        
        return f"""Analyze the cancer mutation {gene}:{variant} and provide a clinical assessment.{context_hint}

Provide your analysis in the following JSON format:
{{
    "pathogenicity": <float 0.0-1.0>,
    "cancer_types": ["<list of associated cancer types>"],
    "protein_change": "<protein change notation>",
    "mechanism": "<molecular mechanism of the mutation>",
    "significance": "<PATHOGENIC/LIKELY_PATHOGENIC/UNCERTAIN/LIKELY_BENIGN/BENIGN>",
    "therapies": ["<list of targeted therapies>"],
    "prognosis": "<poor/moderate/good/uncertain>",
    "clinical_context": "<clinical interpretation and relevance>",
    "confidence": <float 0.0-1.0>
}}

IMPORTANT: Your entire response must be only the JSON object, starting with {{ and ending with }}. Do not include any text before or after the JSON."""
    
    async def _generate(self, session: aiohttp.ClientSession, prompt: str, retry_count: int = 0) -> Optional[str]:
        """Generate response from Ollama with retry logic"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",  # Force JSON mode to prevent conversational text
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
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout for complex requests
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Log performance metrics
                    if 'eval_count' in result and 'eval_duration' in result:
                        tokens = result['eval_count']
                        duration_sec = result['eval_duration'] / 1e9
                        tokens_per_sec = tokens / duration_sec if duration_sec > 0 else 0
                        logger.info(f"Ollama performance: {tokens} tokens in {duration_sec:.2f}s = {tokens_per_sec:.1f} tok/s")
                    
                    return result.get('response', '')
                elif response.status == 500 and retry_count < 2:
                    # Retry on server errors
                    logger.warning(f"Ollama returned 500, retrying... (attempt {retry_count + 2}/3)")
                    await asyncio.sleep(2)  # Wait 2 seconds before retry
                    return await self._generate(session, prompt, retry_count + 1)
                else:
                    logger.error(f"Ollama API returned status {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.error("Ollama request timed out after 180 seconds")
            return None
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None
    
    def _parse_ai_response(self, ai_text: str, gene: str, variant: str) -> Dict:
        """Parse JSON response from AI"""
        try:
            # Try direct JSON parsing first
            if ai_text.strip().startswith('{'):
                try:
                    return json.loads(ai_text.strip())
                except:
                    pass
            
            # Extract JSON from response
            import re
            # Look for JSON between first { and last }
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', ai_text, re.DOTALL)
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
            "PTEN", "APC", "MLH1", "MSH2", "VHL", "RB1", "BRAF"
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
    
    def _parse_multi_mutation_response(self, response: str) -> Dict:
        """Parse AI response for multi-mutation analysis"""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                return json.loads(response)
            
            # Otherwise, structure the response
            return {
                "individual_analyses": self._extract_individual_analyses(response),
                "multi_mutation_insights": self._extract_multi_mutation_insights(response),
                "pathway_interactions": self._extract_pathway_interactions(response),
                "composite_risk": self._extract_composite_risk(response),
                "therapeutic_strategy": self._extract_therapeutic_strategy(response)
            }
        except Exception as e:
            logger.error(f"Failed to parse multi-mutation response: {e}")
            return self._fallback_multi_mutation_analysis()
    
    def _extract_mutations_from_prompt(self, prompt: str) -> List[Tuple[str, str]]:
        """Extract mutations from the prompt"""
        mutations = []
        # Look for patterns like GENE:variant
        import re
        pattern = r'([A-Z0-9]+):\s*([cp]\.[^\s,\]]+)'
        matches = re.findall(pattern, prompt)
        for gene, variant in matches:
            mutations.append((gene, variant))
        logger.info(f"Extracted {len(mutations)} mutations from prompt: {mutations}")
        return mutations
    
    async def _analyze_individual_mutations(self, mutations: List[Tuple[str, str]]) -> Dict:
        """Analyze each mutation individually with Gemma 3n"""
        analyses = {}
        total_mutations = len(mutations)
        
        # Process mutations one at a time to avoid overwhelming Ollama
        for i, (gene, variant) in enumerate(mutations, 1):
            logger.info(f"Gemma 3n analyzing mutation {i}/{total_mutations}: {gene}:{variant}")
            try:
                analysis = await self.analyze_cancer_mutation(gene, variant)
                analyses[f"{gene}:{variant}"] = analysis
                logger.info(f"Completed analysis {i}/{total_mutations}")
                # Small delay between requests to prevent overload
                if i < total_mutations:
                    await asyncio.sleep(1.0)  # Increased delay to prevent timeouts
            except Exception as e:
                logger.error(f"Failed to analyze {gene}:{variant}: {e}")
                # Use fallback for this mutation
                analyses[f"{gene}:{variant}"] = self._fallback_analysis(gene, variant)
        return analyses
    
    def _extract_individual_analyses(self, response: str) -> Dict:
        """Extract individual mutation analyses from response"""
        # This would parse the response to extract individual mutation info
        # For now, return empty dict - would be implemented based on response format
        return {}
    
    def _extract_multi_mutation_insights(self, response: str) -> List[str]:
        """Extract multi-mutation insights from response"""
        insights = []
        
        # Look for interaction patterns
        if "synergistic" in response.lower():
            insights.append("Synergistic pathogenic effects identified between mutations")
        if "pathway convergence" in response.lower():
            insights.append("Multiple mutations converge on common pathways")
        if "synthetic lethality" in response.lower():
            insights.append("Potential synthetic lethality opportunities identified")
        
        return insights
    
    def _extract_pathway_interactions(self, response: str) -> Dict:
        """Extract pathway interaction information from response"""
        return {
            "converging_pathways": [],
            "interaction_type": "unknown",
            "therapeutic_implications": []
        }
    
    def _extract_composite_risk(self, response: str) -> str:
        """Extract composite risk assessment from response"""
        if "high risk" in response.lower():
            return "high"
        elif "moderate risk" in response.lower():
            return "moderate"
        elif "low risk" in response.lower():
            return "low"
        else:
            return "uncertain"
    
    def _extract_therapeutic_strategy(self, response: str) -> Dict:
        """Extract therapeutic strategy from response"""
        return {
            "primary_targets": [],
            "combination_approaches": [],
            "clinical_trials": []
        }
    
    def _fallback_multi_mutation_analysis(self) -> Dict:
        """Fallback analysis for multiple mutations"""
        return {
            "individual_analyses": {},
            "multi_mutation_insights": [
                "Multiple mutations detected - consider molecular tumor board review",
                "Comprehensive genomic profiling recommended"
            ],
            "pathway_interactions": {
                "converging_pathways": [],
                "interaction_type": "unknown",
                "therapeutic_implications": ["Multi-target approach may be beneficial"]
            },
            "composite_risk": "uncertain",
            "therapeutic_strategy": {
                "primary_targets": [],
                "combination_approaches": ["Consider combination targeted therapy"],
                "clinical_trials": ["Search for trials targeting multiple pathways"]
            }
        }