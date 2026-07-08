# Make Python Dataset for Latent Space Alignment (5% Chat, 95%Python)

import json
import datasets
import argparse
import math
import re
import os  # Required for hard termination bypass
import sys

"""
Proportional Meta-Dataset Harvester.
Optimized to handle streaming network threads without core-dumping.
"""

def clean_text(text: str) -> str:
    """Sanitizes text of legacy prompt artifacts and structural leaks."""
    if not isinstance(text, str): return ""
    
    # 1. Scrub the MetaMathQA structural leakage ("#### 123" at the end of strings)
    text = re.sub(r'####\s*-?\\d+(\\.\\d+)?\s*$', '', text).strip()
    
    # 2. Remove hardcoded legacy tags that conflict with downstream dynamic formatters
    legacy_tags = [
        "### Instruction:", "### Response:", "### Input:", 
        "USER:", "ASSISTANT:", "System:", "Human:", "Assistant:"
    ]
    for tag in legacy_tags:
        text = text.replace(tag, "").strip()
        
    return text.strip()

def parse_polymorphic_row(x: dict) -> dict:
    """Bulletproof polymorphic parser that auto-detects fields and text structures."""
    if not isinstance(x, dict): return {"instruction": "", "output": ""}
    
    # 1. Check for list-like conversation structures (conversations, messages, turns, etc.)
    for key in ["conversations", "messages", "turns", "history"]:
        if key in x and isinstance(x[key], list) and len(x[key]) > 0:
            instr, out = "", ""
            for turn in x[key]:
                if not isinstance(turn, dict): continue
                role = (turn.get("role") or turn.get("from") or turn.get("type") or "").lower()
                content = turn.get("content") or turn.get("value") or turn.get("text") or ""
                
                if role in ["user", "human", "client"]:
                    if not instr: instr = content
                elif role in ["assistant", "gpt", "bot", "response"]:
                    if not out: out = content
            if instr and out:
                return {"instruction": instr, "output": out}

    # 2. Check for direct flat text/DPO alignment mappings
    pairs = [
        ("fr_question", "fr_deepseek_attempt"), # Localized High-Fidelity French CoT Tracks
        ("prompt", "accepted_completion"),       # Specific to legmlai/openhermes-fr
        ("instruction", "response"),
        ("instruction", "output"),
        ("question", "solution"),
        ("question", "answer"),
        ("prompt", "chosen"),
        ("prompt", "response"),
        ("query", "response"),
        ("query", "answer"),
        ("question", "response")
    ]
    for k_in, k_out in pairs:
        if k_in in x and k_out in x:
            val_out = x[k_out]
            if isinstance(val_out, list) and len(val_out) > 0 and isinstance(val_out[-1], dict):
                val_out = val_out[-1].get("content") or val_out[-1].get("value") or ""
            if x[k_in] and val_out:
                return {"instruction": str(x[k_in]), "output": str(val_out)}
                
    # 3. Flat text fallback
    if "text" in x and isinstance(x["text"], str) and x["text"]:
        text = x["text"]
        if "### Instruction:" in text and "### Response:" in text:
            parts = text.split("### Response:")
            return {"instruction": parts[0].replace("### Instruction:", "").strip(), "output": parts[1].strip()}
            
    return {"instruction": "", "output": ""}

def main():
    parser = argparse.ArgumentParser(description="Bilingual Proportional Meta-Dataset Harvester")
    parser.add_argument("--total_samples", type=int, default=2000, help="Total samples to harvest.")
    parser.add_argument("--output", type=str, default="bridge_bilingual_mixture.json", help="Output JSON file name.")
    parser.add_argument("--max_chars", type=int, default=4000, help="Max length per sequence to prevent OOM.")
    parser.add_argument("--fr_omnibus_repo", type=str, default="legmlai/openhermes-fr", 
                        help="HuggingFace French Omnibus dataset identifier.")
    args = parser.parse_args()

    # Balanced Mixture Grid: 50% English / 50% French Symmetric Distribution
    META_HARVEST_CONFIGS = [
    # --- GENERAL CONVERSATIONAL ANCHOR (5% Total Weight) ---
    # Objective: Maintain conversational stability and markdown formatting syntax.
    {
        "id": "HuggingFaceH4/ultrachat_200k", 
        "split": "train_sft", 
        "weight": 0.05, 
        "name": "English Chat",
        "map_fn": lambda x: {
            "instruction": x.get("messages", [{"content": ""}])[0].get("content", ""),
            "output": x.get("messages", [{"content": ""}, {"content": ""}])[1].get("content", ""),
            "meta": "EN-Chat"
        }
    },
    
    # --- PURE PYTHON EXPERIMENTAL TRACKS (95% Total Weight) ---
    # Objective: Isolate structural code layout to validate the geometric manifold.
    {
        "id": "iamtarun/python_code_instructions_18k_alpaca", 
        "split": "train", 
        "weight": 0.32, 
        "name": "Python Alpaca Instructions",
        "map_fn": lambda x: {
            "instruction": f"{x.get('instruction', '')}\n\nInput Context:\n{x.get('input', '')}".strip() if x.get('input') and x.get('input').strip() else x.get('instruction', ''),
            "output": x.get("output", ""), 
            "meta": "PY-Alpaca-Basic"
        }
    },
    {
        "id": "Vezora/Tested-22k-Python-Alpaca", 
        "split": "train", 
        "weight": 0.32, 
        "name": "Python Tested Production",
        "map_fn": lambda x: {
            "instruction": f"{x.get('instruction', '')}\n\nInput Context:\n{x.get('input', '')}".strip() if x.get('input') and x.get('input').strip() else x.get('instruction', ''),
            "output": x.get("output", ""), 
            "meta": "PY-Tested-Advanced"
        }
    },
    {
        "id": "mlabonne/Evol-Instruct-Python-26k", 
        "split": "train", 
        "weight": 0.31, 
        "name": "Python Evol Complex",
        "map_fn": lambda x: {
            "instruction": x.get("instruction", ""), 
            "output": x.get("output", ""), 
            "meta": "PY-Evol-Complex"
        }
    }
]
    harvested_matrix = []
    print(f"Initializing Proportional Harvester...")
    print(f"Target Total: {args.total_samples} samples | Max Chars: {args.max_chars}\n")

    for config in META_HARVEST_CONFIGS:
        target_samples = math.ceil(args.total_samples * config["weight"])
        print(f" ➔ Track: {config['name']:<18} | Target: {target_samples}")
        
        try:
            load_kwargs = {"split": config["split"], "streaming": True}
            if "config_name" in config:
                load_kwargs["name"] = config["config_name"]
                
            ds = datasets.load_dataset(config["id"], **load_kwargs)
            iterator = iter(ds)
            count = 0
            
            rows_examined = 0
            keys_seen = None
            skips_empty = 0
            skips_length = 0
            
            while count < target_samples:
                try:
                    row = next(iterator)
                    rows_examined += 1
                    if keys_seen is None:
                        keys_seen = list(row.keys())
                        
                    if "filter_fn" in config and not config["filter_fn"](row):
                        continue
                        
                    parsed = config["map_fn"](row)
                    instr = clean_text(parsed.get("instruction", ""))
                    out = clean_text(parsed.get("output", ""))
                    
                    if not instr or not out:
                        skips_empty += 1
                        continue
                    if len(instr) > args.max_chars or len(out) > args.max_chars:
                        skips_length += 1
                        continue
                    
                    harvested_matrix.append({
                        "id": f"{config['name'].lower().replace(' ', '_')}_{count}",
                        "domain": parsed["meta"],
                        "instruction": instr,
                        "output": out
                    })
                    count += 1
                except StopIteration:
                    print(f"      ! Reached end of dataset early.")
                    break
            
            print(f"   ✓ Harvested {count} records.")
            
            # Proactive cleanup of specific generator references to close streaming sockets
            del iterator
            del ds
            
            if count == 0 and rows_examined > 0:
                print(f"        Telemetry Diagnostic for [{config['name']}]:")
                print(f"        - Total rows streamed and examined: {rows_examined}")
                print(f"        - Available schema keys found: {keys_seen}")
                print(f"        - Skipped due to parsing failure / empty fields: {skips_empty}")
                print(f"        - Skipped due to exceeding max_chars ({args.max_chars}): {skips_length}")
                
        except Exception as e:
            print(f"   ✗ Error processing {config['id']}: {e}")

    harvested_matrix = harvested_matrix[:args.total_samples]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(harvested_matrix, f, indent=4, ensure_ascii=False)

    print(f"\nBalanced Mixture Complete. Generated '{args.output}' containing {len(harvested_matrix)} balanced rows.")
    
    # --- HARD FINALIZATION BYPASS ---
    # Manually flush terminal buffers and cut the process cleanly.
    # This completely prevents multi-threaded background network retries from clashing with Python GIL destruction.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
