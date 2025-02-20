import os

from predict import *
from scorer import *
from datetime import datetime

def run_exp_refine_marks(retrieval_file, prompt_file, prompt_refine_marks_file, pred_folder, ref_dicts, scores_file, temperatures, prompt_params, exp_name, refine_type='score'):
    """
    run multiple prompt templates
    """
    print('START OF EXPERIMENT')
    # read model_input and docs from retriever
    # features = ['id', 'model_input', 'model_output_text'] + [f'doc_{l}' for l in range(1, n_docs+1)]
    df = pd.read_json(retrieval_file, lines=True)#[features]
    #print(ref_dicts)
    # read prompt
    with open(prompt_file, 'r') as f:
        prompt_templates = f.read().split(',\n\n\n')
    with open(prompt_refine_marks_file, 'r') as f:
        prompt_refine_marks_templates = f.read().split(',\n\n\n')
    #print(df['id'])
    for t in [float(temp) for temp in temperatures.split(',')]:
        
        for prompt_num, prompt_template in enumerate(prompt_templates):
            print(f'PROMPT #{prompt_num}, T={t}')
            # get qwen responce
            df['mark'] = 0
            for i in tqdm(df.index):
                ref_index = [k for k in range(len(ref_dicts)) if ref_dicts[k]['id'] == df.loc[i, 'id']][0]
                #print(ref_index)
                print(ref_index)
                #print(ref_dicts[i])
                print(f"Correct answer: [{[df.loc[i, 'model_output_text'][ref_dicts[ref_index]['hard_labels'][j][0]:ref_dicts[ref_index]['hard_labels'][j][1]] for j in range(len(ref_dicts[ref_index]['hard_labels']))]}]")
                params={'temperature':t}
                if prompt_params != '':
                    for p in prompt_params.split(','):
                        params[p] = df.loc[i, p]
                #### add your own params for promt here like:
                # params['your_parameter'] = some_func(df.loc[i, 'param_name_in_json_file'])
                
                ####
                df.loc[i, 'response'] = get_qwen_response(prompt_template, **params)
                if refine_type == 'score':
                    df.loc[i, 'mark'] = get_qwen_response_refine(df.loc[i,:], prompt_refine_marks_templates[-1], **params, T=t)
                elif refine_type == 'regenerate':
                    df.loc[i, 'response'] = get_qwen_response_refine(df.loc[i,:], prompt_refine_marks_templates[-1], **params, T=t)
            # convert responces to predictions
            forward_features = ['id', 'hard_labels', 'model_input', 'model_output_text', 'response']
            #df.to_json(f'hallucinations_with_marks/detection_with_marks_{datetime.now().isoformat()}.json', orient='records')
            if 'hard_labels' in df.columns:
                df['golden_hard_labels'] = df.hard_labels.copy()
                forward_features += ['golden_hard_labels']
            if 'soft_labels' in df.columns:
                df['golden_soft_labels'] = df.soft_labels.copy()
                forward_features += ['golden_soft_labels']
            model_responses = process_model_response(df, forward_features=forward_features)
            # write to json
            pred_file=os.path.join(pred_folder, f"prompt_{prompt_num}_predictions.json")
            model_responses.to_json(pred_file, lines=True, orient='records')
            print(ref_dicts)
            print(load_jsonl_file_to_records(pred_file, is_ref=False))
            # count metrics
            get_scores_refine(
                df=df,
                prompt=prompt_template,
                refine_prompt=prompt_refine_marks_templates[-1],
                ref_dicts=ref_dicts, 
                pred_dicts=load_jsonl_file_to_records(pred_file, is_ref=False), 
                comment=f'exp "{exp_name}":prompt # {prompt_num} from {prompt_file}', 
                output_file=scores_file,
                temp=t,
                refine_type=refine_type)
    print('end OF EXPERIMENT')


if __name__ == '__main__':
    p = ap.ArgumentParser()
    # cmd args
    p.add_argument('--prompt-file', type=str, help='text file containing prompt templates, separated with "<sep>"')
    p.add_argument('--prompt-refine-marks-file', type=str, help='text file containing prompt templates, separated with "<sep>"')
    p.add_argument('--retrieval-file', type=str, help='json file containing model_input and doc_i')
    p.add_argument('--pred-folder', type=str, default='predictions', help='json file to write prediction')
    p.add_argument('--ref-file', type=str)
    p.add_argument('--scores-file', type=str, help='text file to contain scores')
    p.add_argument('--temperatures', type=str, default="1.0", help='float list temperatures of generation separated by "," (ex: "1.0,2.0")')
    p.add_argument('--prompt-params', type=str, default="doc_1,doc_2,doc_3,doc_4,doc_5,model_input,model_output_text", help='str list of params for prompt separated by "," (ex: "doc_1,doc_2")')
    p.add_argument('--exp-name', type=str, default="default")
    p.add_argument('--refine-type', type=str, default='score')
    a = p.parse_args()
    
    run_exp_refine_marks(
        retrieval_file=a.retrieval_file, 
        prompt_file=a.prompt_file, 
        prompt_refine_marks_file=a.prompt_refine_marks_file,
        pred_folder=a.pred_folder,
        ref_dicts=load_jsonl_file_to_records(a.ref_file), 
        scores_file=a.scores_file,
        temperatures=a.temperatures,
        prompt_params=a.prompt_params,
        exp_name=a.exp_name,
        refine_type=a.refine_type
    )

