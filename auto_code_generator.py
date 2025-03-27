import torch
import re
import os

# prompt generating systematically
def generate_prompt(result, max_chunk_len=30):
    summary = result.summary
    # input_text = result.text
    # chunks = input_text.split('\n')
    # useful = ""
    # patterns = [
    #     r'step\s*\d+',
    #     r'\bdef\s+\w+\s*\(',  # python-style function
    #     r'\b(input|output|return|initialize|loop|while|for|if)\b',
    #     r'=>|:=|==|->|=',   # symbolic logic
    #     r'loss function|gradient|activation|architecture|layer',
    #     r'\b[a-zA-Z_]+\s*=\s*',  # assignment
    # ]
    # patterns_combined = '(?i)'+'|'.join(patterns)
    # for chunk in chunks:
    #     if re.search(patterns_combined, chunk):
    #         if len(chunk)>max_chunk_len:
    #             useful += '\n'+chunk[:max_chunk_len]
    #         else:
    #             useful += '\n'+chunk
                
    
    prompt = "Generate Python code:\n" + summary
    return prompt
    

# expect the fine-tuned pretrained model - see model_comparison.ipynb file for detail
def generate_code(prompt, model, tokenizer, max_len=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    encoded = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)


    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_len,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output[len(prompt):]


def save_code(code, filename):
    downloads_path = os.path.expanduser("~/Downloads/")
    os.makedirs(downloads_path, exist_ok=True)
    file_path = os.path.join(downloads_path, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"Generated Code has been saved to {file_path}")