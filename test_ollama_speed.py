#!/usr/bin/env python3
"""
Ollama Speed Test for OncoScope Gemma 3n Model
Tests inference speed, tokens/second, and response times
"""

import time
import requests
import json
import statistics
from typing import List, Dict

MODEL_NAME = "oncoscope-cancer:latest"
OLLAMA_URL = "http://localhost:11434"

# Test mutations of varying complexity
TEST_MUTATIONS = [
    {"gene": "BRAF", "variant": "c.1799T>A", "name": "BRAF V600E (simple)"},
    {"gene": "TP53", "variant": "c.524G>A", "name": "TP53 (complex)"},
    {"gene": "EGFR", "variant": "c.2369C>T", "name": "EGFR T790M"},
]

def test_single_mutation(gene: str, variant: str) -> Dict:
    """Test a single mutation and return timing metrics"""
    
    prompt = f"""Analyze the cancer mutation {gene}:{variant} and provide a clinical assessment.

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

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 1024,
            "seed": 42
        }
    }
    
    # Time the request
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=300
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract metrics
            metrics = {
                "success": True,
                "elapsed_time": elapsed,
                "response_length": len(data.get("response", "")),
                "total_duration": data.get("total_duration", 0) / 1e9,  # Convert nanoseconds to seconds
                "load_duration": data.get("load_duration", 0) / 1e9,
                "prompt_eval_duration": data.get("prompt_eval_duration", 0) / 1e9,
                "eval_duration": data.get("eval_duration", 0) / 1e9,
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_count": data.get("eval_count", 0),
            }
            
            # Calculate tokens per second
            if metrics["eval_duration"] > 0:
                metrics["tokens_per_second"] = metrics["eval_count"] / metrics["eval_duration"]
            else:
                metrics["tokens_per_second"] = 0
                
            if metrics["prompt_eval_duration"] > 0:
                metrics["prompt_tokens_per_second"] = metrics["prompt_eval_count"] / metrics["prompt_eval_duration"]
            else:
                metrics["prompt_tokens_per_second"] = 0
                
            return metrics
            
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "elapsed_time": elapsed
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Timeout",
            "elapsed_time": time.time() - start_time
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }

def run_speed_tests(num_runs: int = 3):
    """Run multiple speed tests and calculate statistics"""
    
    print(f"🧬 OncoScope Ollama Speed Test")
    print(f"📊 Model: {MODEL_NAME}")
    print(f"🔄 Runs per mutation: {num_runs}")
    print("=" * 60)
    
    # Check if model is available
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = [m.get("name", "") for m in response.json().get("models", [])]
            if MODEL_NAME not in " ".join(models):
                print(f"⚠️  Warning: Model '{MODEL_NAME}' not found in Ollama")
                print(f"   Available models: {', '.join(models)}")
        else:
            print("⚠️  Warning: Could not connect to Ollama")
    except:
        print("⚠️  Warning: Ollama may not be running")
    
    print()
    
    all_results = []
    
    for mutation in TEST_MUTATIONS:
        print(f"\n🧪 Testing: {mutation['name']}")
        print("-" * 40)
        
        mutation_results = []
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}: ", end="", flush=True)
            
            result = test_single_mutation(mutation["gene"], mutation["variant"])
            mutation_results.append(result)
            
            if result["success"]:
                print(f"✅ {result['elapsed_time']:.2f}s ({result['tokens_per_second']:.1f} tok/s)")
            else:
                print(f"❌ {result['error']} ({result['elapsed_time']:.2f}s)")
        
        # Calculate statistics for successful runs
        successful_runs = [r for r in mutation_results if r["success"]]
        
        if successful_runs:
            avg_time = statistics.mean([r["elapsed_time"] for r in successful_runs])
            avg_tokens_per_sec = statistics.mean([r["tokens_per_second"] for r in successful_runs])
            avg_eval_time = statistics.mean([r["eval_duration"] for r in successful_runs])
            
            print(f"\n  📊 Statistics:")
            print(f"     Success rate: {len(successful_runs)}/{num_runs}")
            print(f"     Avg total time: {avg_time:.2f}s")
            print(f"     Avg inference time: {avg_eval_time:.2f}s")
            print(f"     Avg tokens/sec: {avg_tokens_per_sec:.1f}")
            
            if len(successful_runs) > 1:
                std_time = statistics.stdev([r["elapsed_time"] for r in successful_runs])
                print(f"     Std deviation: {std_time:.2f}s")
        
        all_results.extend(mutation_results)
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("📈 OVERALL STATISTICS")
    print("=" * 60)
    
    successful_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]
    
    if successful_results:
        print(f"✅ Success rate: {len(successful_results)}/{len(all_results)} ({len(successful_results)/len(all_results)*100:.1f}%)")
        print(f"⏱️  Average total time: {statistics.mean([r['elapsed_time'] for r in successful_results]):.2f}s")
        print(f"🚀 Average tokens/sec: {statistics.mean([r['tokens_per_second'] for r in successful_results]):.1f}")
        print(f"📝 Average response length: {statistics.mean([r['response_length'] for r in successful_results]):.0f} chars")
        
        # Breakdown by operation
        print(f"\n🔍 Time Breakdown (averages):")
        print(f"   Model loading: {statistics.mean([r['load_duration'] for r in successful_results]):.2f}s")
        print(f"   Prompt processing: {statistics.mean([r['prompt_eval_duration'] for r in successful_results]):.2f}s")
        print(f"   Generation: {statistics.mean([r['eval_duration'] for r in successful_results]):.2f}s")
    
    if failed_results:
        print(f"\n❌ Failures: {len(failed_results)}")
        failure_reasons = {}
        for r in failed_results:
            reason = r.get("error", "Unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        for reason, count in failure_reasons.items():
            print(f"   - {reason}: {count}")
    
    # Performance recommendations
    print("\n" + "=" * 60)
    print("💡 PERFORMANCE INSIGHTS")
    print("=" * 60)
    
    if successful_results:
        avg_time = statistics.mean([r['elapsed_time'] for r in successful_results])
        avg_tokens = statistics.mean([r['tokens_per_second'] for r in successful_results])
        
        if avg_time > 30:
            print("⚠️  Response times are quite high (>30s)")
            print("   Consider:")
            print("   - Using a quantized model (Q4 instead of Q8)")
            print("   - Reducing max tokens (num_predict)")
            print("   - Running on GPU if currently on CPU")
        elif avg_time > 10:
            print("⚡ Response times are moderate (10-30s)")
            print("   For better UX, consider Q4 quantization")
        else:
            print("✅ Response times are good (<10s)")
        
        if avg_tokens < 10:
            print("\n⚠️  Token generation is slow (<10 tok/s)")
            print("   This suggests CPU inference or memory constraints")
        elif avg_tokens < 50:
            print("\n⚡ Token generation is moderate (10-50 tok/s)")
        else:
            print("\n✅ Token generation is fast (>50 tok/s)")

if __name__ == "__main__":
    # You can adjust the number of runs
    run_speed_tests(num_runs=3)