from transformers import AutoTokenizer
from exa_search import search_papers, output_final_result, save_analysis
from auto_code_generator import generate_prompt, generate_code, save_code
from load_model import load_encoder_model, load_finetuned_codegen_model

import sys

def get_search_params():
    query = input("\nRT Research Assistant: Enter ML research topic:\n>>>")
    num_results = input("\nRT Research Assistant: Enter the number of papers to result out:\n>>>")
    if not num_results:
        num_results = '3'
    while not num_results.isdigit():
        print("\nRT Research Assistant: Invalid input, enter integer number.")
        num_results = input("RT Research Assistant: Enter the number of papers to result out:\n>>>")
        if not num_results:
            num_results = '3'
    num_results = int(num_results)
    return query, num_results


def get_filename_analysis():
    filename = input("RT Research Assistant: Input filename without file type:\n>>>")
    filename += ".json"
    return filename


def search_and_analyze():
    query, num_results = get_search_params()
    print("\nRT Research Assistant: Searching for relevant papers...")
    results = search_papers(query, num_results)
    if not results:
        print("\nRT Research Assistant: No relevant paper found.")
        cont = input("RT Research Assistant: Type q to quit, any other values to start over.\n>>>")
        if cont.lower() == 'q':
            print("\nRT Research Assistant: Exiting...")
            sys.exit()
        else:
            return search_and_analyze()
            
    results = results.results
    
    # load model and tokenizer after papers found
    print("Analyzing...")
    model = load_encoder_model()  # alredy set to eval mode
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    # final result
    rel_scores = output_final_result(query, results, model, tokenizer)
    
    # save
    to_save = input("\nRT Research Assistant: Do you want to save the results? [y|n]\n>>>")
    if to_save == 'y':
        while True:
            filename = get_filename_analysis()
            try:
                save_analysis(query, results, rel_scores, filename)
                break
            except:
                print("RT Research Assistant: Failed to save the analysis. This is most likely because of the invalid filename.")
                get_filename_analysis()
    
    return results


def code_generation(results):
    code_gen = input("\nRT Research Assistant: Do you want to proceed code generation? [y|n]\n>>>")
    if code_gen=='y':
        idx = input(f"RT Research Assistant: Which research do you want to generate code? Type index from 1-{len(results)}.\n>>>")
        if not idx:
            idx = '1'
        while not idx.isdigit():
            print("\nRT Research Assistant: Invalid input, enter integer number.")
            idx = input(f"RT Research Assistant: Which research do you want to generate code? Type index from 1-{len(results)}.\n>>>")
            if not idx:
                idx = '1'
        idx = int(idx)
        result = results[idx-1]
        
        print("\nGenerating code...\n")
        tokenizer, model = load_finetuned_codegen_model()
        prompt = generate_prompt(result)
        generated_code = generate_code(prompt, model, tokenizer)
        
        print(f"Prompt:\n{prompt}\n")
        print(f"Code:\n{generated_code}")
        save_generated_code_opt = input("\nRT Research Assistant: Do you want to save the generated code in a text file (.txt)? [y|n]\n>>>")
        if save_generated_code_opt=='y':
            def get_filename_gencode():
                filename = input("RT Research Assistant: Input filename without file type:\n>>>")
                filename += ".txt"
                return filename
            while True:
                filename = get_filename_gencode()
                try:
                    save_code(generated_code, filename)
                    break
                except:
                    print("RT Research Assistant: Failed to save the code. This is most likely because of the invalid filename.")
                    get_filename_gencode()
    
                        
def main():
    while True:
        results = search_and_analyze()
        if results:
            while True:
                code_generation(results)
                continue_opt = input("\nRT Research Assistant: Select a continue option:\n1 = Generate code for another paper found\n2 = Search papers with a different topic\n3 = Quit\n>>>")
                if continue_opt=='1':
                    continue
                elif continue_opt=='2':
                    break
                else:
                    print("RT Research Assistant: Thank you for using RT Research Assistant. Now exiting...")
                    sys.exit()
                    
        print("Unexpected results.")
    

if __name__ == "__main__":
    main()