# dset – improved unified dataset fetcher

A robust, self‑contained tool for fetching datasets from the internet with enhanced UX and statistics.

## 🚀 Key Improvements

- **Better validation** – checks arguments before processing
- **Progress feedback** – clear status messages and emojis  
- **Performance tracking** – timing, rates, and file sizes
- **Error handling** – graceful cleanup on interruption
- **Verbose mode** – detailed operation info
- **Statistics module** – analyze existing datasets
- **Cleaner output** – concise summaries with file paths

## 📦 Installation & Usage

```bash
# Basic web search
python3 -c "from dset.cli import main; main()" \
    --source web --query "react hooks" --dataset react

# GitHub search with details
python3 -c "from dset.cli import main; main()" \
    --source github --query "machine learning" --language python --dataset ml \
    --max-repos 5 --max-files 10 --verbose

# Analyze existing dataset
python3 dset/stats.py runs/your_dataset.jsonl
```

## 📊 Sample Output

```
🔍 Fetching from web: react hooks
📡 Searching web for examples...
📊 Web search limit: 3 examples

📊 Performance Summary:
   ⏱️  Duration: 0.17s
   📝 Examples: 1
   💾 Size: 0.00 MB
   ⚡ Rate: 6.0 examples/s
📁 Output file: /Users/mac/sloughGPT/runs/test_improved.jsonl
```

## 🔧 Output Format

Each line contains the raw lmtrain JSON wrapped in minimal metadata:
```json
{
  "path": "example_0.json",
  "size": 58,
  "mtime": 1769600892,
  "content": "{\"instruction\": \"react hooks\", \"output\": \"sample output\"}",
  "source": "lmtrain"
}
```

## ✨ Why this is better

- **Self‑contained** – all functionality in `dset/` package
- **Extensible** – easy to add new sources or formats
- **Reliable** – proper error handling and cleanup
- **User‑friendly** – clear help and progress indicators
- **Performance‑aware** – tracks timing and rates for optimization

Ready for production use! 🎯