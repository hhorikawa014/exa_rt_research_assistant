# for the flexibility, using requrests module instead of exa_py
from exa_py import Exa
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

EXA_API_KEY = "Your EXA API key"

"""
General Flow:
    1. call search_papers with query
    2. tokenize the output if it is in the correct format
    3. inference to get model's ouput
    4. generate summary
    5. generate sentences for reference
"""

def search_papers(query: str, num_results: int=3):

    exa = Exa(api_key = EXA_API_KEY)

    result = exa.search_and_contents(
    query,
    text = True,
    category = "research paper",
    num_results = num_results,
    livecrawl = "always",
    summary = {"query": "In only one sentence, create a prompt for code implementation about the paper."}
    )
    
    return result
    

# tokenizing input text
def tokenize(texts, tokenizer, seq_len=512):
    tokenized_texts = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    return tokenized_texts["input_ids"]


# get model output using input_ids
def inference(model, input_ids, mask):
    with torch.no_grad():
        output = model(input_ids, mask)
    return output
    
    
# summarize the encoder output using decoder
def summarize(model, source_input, tokenizer, max_len: int=200, k: int=5):
    source_mask = (source_input != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
    model_output = inference(model, source_input, source_mask)
    
    # using attention-weighted selection to generate summarization output
    attention_scores = torch.mean(model_output, dim=-1)
    top_k_indices = attention_scores.topk(k=k, dim=-1).indices
    summary_output = [tokenizer.decode(source_input[i, top_k_indices[i][:max_len]].tolist(), skip_special_tokens=True) for i in range(source_input.shape[0])]
    return summary_output
    
def output_final_result(query, results, model, tokenizer):
    print("Calculating reliability score...")
    rel_scores, topic_groups = compute_reliability_score_and_topics(results, model, tokenizer)
    print("\n*** Research Summary ***")
    print(f"Query: {query}")
    for i, result in enumerate(results):
        result.score = result.score if (isinstance(result.score, float) or isinstance(result.score, int)) else 0
        print(f"\n\nPaper {i+1}:\nTitle: {result.title}\nSummary: {result.summary}\nRelevant Score: {round(100.0*result.score, 1)}\nReliability Score: {rel_scores[i]}\nReference: {result.url}")
    # print("\nGraph:")
    # G = knowledge_graph(results, topic_groups)
    # visualize_graph(G)
    return rel_scores
        
        
def compute_reliability_score_and_topics(results, model, tokenizer):
    papers = [results[i].text for i in range(len(results))]
    
    tokenized_papers = tokenize(papers, tokenizer)
    source_mask = (tokenized_papers != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
    model_output = inference(model, tokenized_papers, source_mask)
    norm_scores = model_output.norm(dim=-1).mean(dim=-1).cpu().numpy()
    sentence_var = model_output.var(dim=-1).mean(dim=-1).cpu().numpy()
    coherence_scores = 1/(1+sentence_var)
    
    scores = (norm_scores+coherence_scores)/2
    scores = [int(score) for score in scores]
    
    topic_groups = []
    for paper in papers:
        tokenized_paper = tokenize(paper, tokenizer)
        source_mask = (tokenized_paper != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        model_output = inference(model, tokenized_paper, source_mask)
        
        tok_importance = model_output.norm(dim=-1).mean(dim=0).cpu().numpy()
        token_ids = tokenized_paper.cpu().numpy()
        words = tokenizer.convert_ids_to_tokens(token_ids[0])
        words = [w for w in words if w not in tokenizer.all_special_tokens and w.isalpha()]
        top_words = np.argsort(tok_importance)[-3:][::-1]  # get top 3 tokens
        topics = [words[i] for i in top_words if i<len(words)]
        topic_groups.append(topics)
    
    return scores, topic_groups
 
 
# def knowledge_graph(results, topic_groups):
#     papers = [results[i].text for i in range(len(results))]
#     G = nx.Graph()
#     for topic_group in topic_groups:
#         for topic in topic_group:
#             G.add_node(topic, type="topic")
#     for i, paper in enumerate(papers):
#         paper_idx = f"Paper {i+1}"
#         G.add_node(paper_idx, type="paper")
#         for topic in topic_groups[i]:
#             G.add_edge(paper_idx, topic)
    
#     return G

# def visualize_graph(G: nx.Graph):
#     plt.figure(figsize=(8,8))
#     pos = nx.spring_layout(G)
#     paper_nodes = [node for node, attr in G.nodes(data=True) if attr.get("type") == "paper"]
#     topic_nodes = [node for node, attr in G.nodes(data=True) if attr.get("type") == "topic"]

#     nx.draw_networkx_nodes(G, pos, nodelist=paper_nodes, node_color="red", node_size=2000)
#     nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_color="lightblue", node_size=1500)
#     nx.draw_networkx_edges(G, pos, edge_color="gray")
#     nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
#     plt.title("Knowledge Graph on Results")
#     plt.show()
    
    
def save_analysis(query, results, reliability_scores, filename):
    analyses = {}
    analyses["query"] = query
    analyses["results"] = []
    for i, result in enumerate(results):
        analysis = {}
        analysis["idx"] = i+1
        analysis["title"] = result.title
        analysis["summary"] = result.summary
        analysis["relevant_score"] = result.score
        analysis["reliability_score"] = reliability_scores[i]
        analysis["reference"] = result.url
        analyses["results"].append(analysis)
    
    downloads_path = os.path.expanduser("~/Downloads/")
    os.makedirs(downloads_path, exist_ok=True)
    file_path = os.path.join(downloads_path, filename)
    with open(file_path, 'w') as f:
        json.dump(analyses, f, indent=4)
    
    print(f"Results have been saved to {file_path}")
    