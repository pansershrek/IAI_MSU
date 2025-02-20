import os

from predict import *
from scorer import *


def run_exp(retrieval_file, prompt_file, pred_folder, ref_dicts, scores_file, temperatures, prompt_params, exp_name):
    """
    run multiple prompt templates
    """
    print('START OF EXPERIMENT')
    # read model_input and docs from retriever
    # features = ['id', 'model_input', 'model_output_text'] + [f'doc_{l}' for l in range(1, n_docs+1)]
    df = pd.read_json(retrieval_file, lines=True)#[features]
    response_col = "response"
    while response_col in df.columns:
        response_col += "_"
    # read prompt
    with open(prompt_file, 'r') as f:
        prompt_templates = f.read().split('<sep>')
    for t in [float(temp) for temp in temperatures.split(',')]:
        for prompt_num, prompt_template in enumerate(prompt_templates):
            print(f'PROMPT #{prompt_num}, T={t}')
            
            # get qwen responce
            for i in tqdm(df.index):
                params={'temperature':t}
                if prompt_params != '':
                    for p in prompt_params.split(','):
                        params[p] = df.loc[i, p]
                #### add your own params for promt here like:
                # params['your_parameter'] = some_func(df.loc[i, 'param_name_in_json_file'])
                
                ####
                df.loc[i, response_col] = get_qwen_response(prompt_template, **params)
            # convert responces to predictions
            forward_features = ['id', 'hard_labels', 'model_input', 'model_output_text', response_col, 'doc_1']
            if 'hard_labels' in df.columns:
                df['golden_hard_labels'] = df.hard_labels.copy()
                forward_features += ['golden_hard_labels']
            if 'soft_labels' in df.columns:
                df['golden_soft_labels'] = df.soft_labels.copy()
                forward_features += ['golden_soft_labels']
            model_responses = process_model_response(df, forward_features=forward_features)
            # write to json
            t_ = str(t).replace('.', "_")
            pred_file=os.path.join(pred_folder, f"prompt_{prompt_num}_predictions_t_{t_}.json")
            model_responses.to_json(pred_file, lines=True, orient='records')
            # count metrics

            get_scores(
                prompt_template,
                ref_dicts=ref_dicts, 
                pred_dicts=load_jsonl_file_to_records(pred_file, is_ref=False), 
                comment=f'exp "{exp_name}":prompt # {prompt_num} from {prompt_file}', 
                output_file=scores_file,
                temp=t)
    print('end OF EXPERIMENT')


if __name__ == '__main__':
    p = ap.ArgumentParser()
    # cmd args
    p.add_argument('--prompt-file', type=str, help='text file containing prompt templates, separated with "<sep>"')
    p.add_argument('--retrieval-file', type=str, help='json file containing model_input and doc_i')
    p.add_argument('--pred-folder', type=str, default='predictions', help='json file to write prediction')
    p.add_argument('--ref-file', type=str)
    p.add_argument('--scores-file', type=str, help='text file to contain scores')
    p.add_argument('--temperatures', type=str, default="1.0", help='float list temperatures of generation separated by "," (ex: "1.0,2.0")')
    p.add_argument('--prompt-params', type=str, default="doc_1,doc_2,doc_3,doc_4,doc_5,model_input,model_output_text", help='str list of params for prompt separated by "," (ex: "doc_1,doc_2")')
    p.add_argument('--exp-name', type=str, default="default")
    a = p.parse_args()

    run_exp(
        retrieval_file=a.retrieval_file, 
        prompt_file=a.prompt_file, 
        pred_folder=a.pred_folder,
        ref_dicts=load_jsonl_file_to_records(a.ref_file), 
        scores_file=a.scores_file,
        temperatures=a.temperatures,
        prompt_params=a.prompt_params,
        exp_name=a.exp_name
    )

