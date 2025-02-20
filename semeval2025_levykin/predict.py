import pandas as pd
import argparse as ap
import requests
from tqdm import tqdm

from difflib import SequenceMatcher

def longest_common_substring(string1, string2):
    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    return match.a, match.a + match.size

def get_qwen_response(prompt_template, T=1, **params):
    # params = {f"doc_{j}": features[f'doc_{j}'] for j in range(1,n_docs+1)}
    # params['model_input'] = features['model_input']
    # params['model_output_text'] = features['model_output_text']
    # print(prompt_template)
    #print(params)
    for i in range(1, 6):
        fin = params[f'doc_{i}'].find('\n\nReferences')
        params[f'doc_{i}'] = params[f'doc_{i}'][:fin].replace('Title:\n', 'Title: ').replace('Content:\n', 'Content:\n').replace('\n\n\n', '\n').replace('\n\n', '\n')
    #print(prompt_template)
    prompt = eval(prompt_template.format(**params))
    print(prompt)
    #print(T)
    r = requests.post('http://10.11.12.131:8080', json={'messages': prompt, 'temperature': T})
    if r.status_code == 200:
        res = r.json()
        if 'result' in res:
            print(res['result'])
            return res['result']
            
    return " "

def get_qwen_response_refine(df, prompt_template, **params):
    #print(df)
    #print(prompt)
    #print(**params)
    prompt = eval(prompt_template.format(doc_1=df['doc_1'], model_input=df['model_input'], model_output_text=df['model_output_text'], 
                                        detected_hallucinations=df['response'][df['response'].find('['):df['response'].find(']')+1]))
    #print(prompt)
    #print(T)
    r = requests.post('http://10.11.12.131:8080', json={'messages': prompt, 'temperature': params['T']})
    if r.status_code == 200:
        res = r.json()
        if 'result' in res:
            print(res['result'])
            return res['result']

def get_hard_labels_old(model_input, model_output, response):
    """
    return: list of int pairs (start, end)
    """
    res = []
    try:
        fragments = eval(response)["hallucinations"]
        for f in fragments:
            f = f.strip().lower()
            s, e = longest_common_substring(model_output.lower(), f)
            if e - s >= max([len(f)*0.75, 1]):
                res += [[s+0, e+0]]
            else:
                print("error hard labels", f)
            # start = model_output.lower().find(f)
            # if start != -1:
            #     end = start + len(f)
            #     res += [[start+0, end+0]]
            # else:
            #     print(f)
        return res
    except:
        print('!')
        return []

from difflib import SequenceMatcher

def find_all_common_substrings(string1, string2):
    """Находит все вхождения подстроки в строке с учетом 75% совпадения"""
    matches = []
    start = 0
    string1, string2 = string1.lower(), string2.lower()
    
    while start < len(string1):
        match = SequenceMatcher(None, string1[start:], string2).find_longest_match(0, len(string1) - start, 0, len(string2))
        
        if match.size >= max(len(string2) * 0.8, 1):
            matches.append((start + match.a, start + match.a + match.size))
            start += match.a + 1  # Двигаем указатель, чтобы найти все вхождения
        else:
            break

    return matches

def get_hard_labels(model_input, model_output, response):
    """
    return: list of int pairs (start, end) for all occurrences
    """
    res = []
    try:
        fragments = eval(response)["hallucinations"]
        for f in fragments:
            f = f.strip().lower()
            matches = find_all_common_substrings(model_output, f)
            if f == 'dna':
                print("MATCHES:", matches)
            if matches:
                res.extend(matches)
            else:
                print("error hard labels", f)
                
        return res
    except:
        print('!')
        return []




def process_model_response(df, forward_features=['id', 'hard_labels']):
    df['hard_labels'] = [[] for _ in range(len(df))]
    for i in df.index:
        labels = get_hard_labels(df.loc[i, 'model_input'], df.loc[i, 'model_output_text'], df.loc[i, 'response'])
        df.loc[i, 'hard_labels'].extend(labels)
    return df[forward_features]


def predict(retrieval_file, prompt_num, prompt_file, pred_file, temperature, prompt_params):
    print('START INFERENCE')
    # read prompt
    with open(prompt_file, 'r') as f:
        prompt_template = f.read().split('<sep>')[prompt_num]
    # read model_input and docs from retriever
    #features = ['id', 'model_input', 'model_output_text'] + [f'doc_{l}' for l in range(1, n_docs+1)]
    df = pd.read_json(retrieval_file, lines=True)#.loc[:, features]
    # get qwen responce
    print('Getting QWEN responses')
    for i in tqdm(df.index):
        
        params={'temperature':temperature}
        params=dict()
        if prompt_params != '':
            for p in prompt_params.split(','):
                params[p] = df.loc[i, p]
        df.loc[i, 'response'] = get_qwen_response(prompt_template, **params)
    # convert responces to predictions
    model_responses = process_model_response(df) # ,forward_features=['id', 'hard_labels', 'model_input', 'model_output', 'response']
    # write to json
    model_responses.to_json(pred_file, lines=True, orient='records')
    print('END OF INFERENCE')


if __name__ == '__main__':
    p = ap.ArgumentParser()
    # cmd args
    p.add_argument('--prompt-file', type=str, help='text file containing prompt templates, separated with "<sep>"')
    p.add_argument('--prompt-num', type=int, default=0, help='int number of prompts in text file, indexing from 0')
    p.add_argument('--retrieval-file', type=str, help='json file containing model_input and doc_i')
    p.add_argument('--pred-file', type=str, default='predictions.json', help='json file to write prediction')
    p.add_argument('--temperature', type=float, default=1.0, help='float temperature of generation')
    p.add_argument('--prompt-params', type=str, default="doc_1,doc_2,doc_3,doc_4,doc_5,model_input,model_output_text", help='str list of params for prompt separated by "," (ex: "doc_1,doc_2")')
    a = p.parse_args()
    predict(
        retrieval_file=a.retrieval_file, 
        prompt_num=a.prompt_num, 
        prompt_file=a.prompt_file, 
        pred_file=a.pred_file, 
        n_docs=a.n_docs,
        temperature=a.temperature,
        prompt_params=a.prompt_params
    )
