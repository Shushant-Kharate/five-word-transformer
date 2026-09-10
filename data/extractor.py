import json

# Absolute paths to your input and output files
input_jsonl_file = r"C:\Users\Shushant\Desktop\5 word transformer\data\Dataset.jsonl"
output_txt_file = r"C:\Users\Shushant\Desktop\5 word transformer\data\targets.txt"

print("Starting extraction...")

# Open the source file to read and the destination file to write
with open(input_jsonl_file, "r", encoding="utf-8") as infile, \
     open(output_txt_file, "w", encoding="utf-8") as outfile:
    
    count = 0
    for line in infile:
        if not line.strip():
            continue
            
        try:
            data = json.loads(line)
            
            # Extract target and save it
            if "target" in data:
                outfile.write(str(data["target"]) + "\n")
                count += 1
                
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")

print(f"Success! Extracted {count} targets into: {output_txt_file}")
