#!/bin/bash

# Quick Ollama Benchmark Script
echo "🚀 Quick Ollama Benchmark for OncoScope"
echo "======================================="

MODEL="oncoscope-gemma-3n"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running. Start it with: ollama serve"
    exit 1
fi

# Simple timing test
echo -e "\n📊 Testing inference speed..."

# Test 1: Short prompt
echo -e "\n1️⃣ Short prompt test (BRAF V600E):"
time curl -s http://localhost:11434/api/generate -d '{
  "model": "'$MODEL'",
  "prompt": "What is BRAF V600E mutation?",
  "stream": false,
  "options": {"num_predict": 100}
}' | jq -r '.total_duration / 1000000000' | xargs -I {} echo "Total time: {} seconds"

# Test 2: Full analysis prompt
echo -e "\n2️⃣ Full analysis test:"
time curl -s http://localhost:11434/api/generate -d '{
  "model": "'$MODEL'",
  "prompt": "Analyze the cancer mutation BRAF:c.1799T>A and provide clinical significance.",
  "stream": false,
  "format": "json",
  "options": {"num_predict": 500}
}' | jq -r '.eval_count, .eval_duration' | paste -sd' ' | awk '{print "Generated " $1 " tokens in " $2/1000000000 " seconds = " $1/($2/1000000000) " tokens/sec"}'

# Test 3: Model info
echo -e "\n3️⃣ Model information:"
curl -s http://localhost:11434/api/show -d '{"name": "'$MODEL'"}' | jq -r '.modelfile' | grep -E "(FROM|PARAMETER)" | head -10

echo -e "\n✅ Benchmark complete!"