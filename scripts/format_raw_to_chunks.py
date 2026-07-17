import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

RAW_FILES = [
    "ct_site.json",
    "ct_services.json",
    "gov_sources.json",
    "industry_blogs.json",
    "qa_pairs.json"
]

EXISTING_FILES = [
    "cleaned_blogs_rag.json",
    "cleaned_rag_testimonials.json"
]

def load_json(filepath: str) -> list:
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")
    data_dir = os.path.join(base_dir, "data")
    
    all_merged_chunks = []
    seen_texts = set()
    duplicates_removed = 0
    short_removed = 0
    
    summary = {}
    
    # Process raw files
    for filename in RAW_FILES:
        filepath = os.path.join(raw_dir, filename)
        chunks = load_json(filepath)
        
        valid_chunks = 0
        for chunk in chunks:
            text = chunk.get("chunk_text", "").strip()
            tokens = chunk.get("chunk_tokens", 0)
            
            if tokens < 15:
                short_removed += 1
                continue
                
            if text in seen_texts:
                duplicates_removed += 1
                continue
                
            seen_texts.add(text)
            all_merged_chunks.append(chunk)
            valid_chunks += 1
            
        summary[filename] = valid_chunks
        
    # Process existing files
    for filename in EXISTING_FILES:
        filepath = os.path.join(data_dir, filename)
        chunks = load_json(filepath)
        
        valid_chunks = 0
        for chunk in chunks:
            text = chunk.get("chunk_text", "").strip()
            if not text:
                # Testimonials use customer_problem, ct_solution, outcome
                if "customer_problem" in chunk:
                    text = f"{chunk.get('customer_problem', '')} {chunk.get('ct_solution', '')} {chunk.get('outcome', '')}".strip()
                else:
                    text = chunk.get("text", "") or chunk.get("content", "")
            
            if text in seen_texts:
                duplicates_removed += 1
                continue
                
            seen_texts.add(text)
            all_merged_chunks.append(chunk)
            valid_chunks += 1
            
        summary[filename] = valid_chunks
        
    # Save output
    output_file = "enriched_knowledge_base.json"
    output_path = os.path.join(data_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_merged_chunks, f, ensure_ascii=False, indent=2)
        
    # Print summary
    print("\n=== Normaliser Summary ===")
    for filename in RAW_FILES + EXISTING_FILES:
        count = summary.get(filename, 0)
        print(f"  {filename:<25}: {count} chunks")
    print("  -------------------------")
    print(f"  Total merged             : {len(all_merged_chunks)} chunks")
    print(f"  Duplicates removed       : {duplicates_removed}")
    print(f"  Short (<15t) removed     : {short_removed}")
    print(f"  Output                   : data/{output_file}")
    
if __name__ == "__main__":
    run()
