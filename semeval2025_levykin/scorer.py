import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from datetime import datetime
import argparse as ap
import json

def recompute_hard_labels(soft_labels):
    """optionally, infer hard labels from the soft labels provided"""
    hard_labels = [] 
    prev_end = -1
    for start, end in (
        (lbl['start'], lbl['end']) 
        for lbl in sorted(soft_labels, key=lambda span: (span['start'], span['end']))
        if lbl['prob'] > 0.5
    ):
        if start == prev_end:
            hard_labels[-1][-1] = end
        else:
            hard_labels.append([start, end])
        prev_end = end
    return hard_labels


def infer_soft_labels(hard_labels):
    """reformat hard labels into soft labels with prob 1"""
    return [
        {
            'start': start,
            'end': end,
            'prob': 1.0,
        }
        for start, end in hard_labels
    ]


def load_jsonl_file_to_records(filename, is_ref=True):
    """read data from a JSONL file and format that as a `pandas.DataFrame`.
    Performs minor format checks (ensures that some labels are present,
    optionally compute missing labels on the fly)."""
    df = pd.read_json(filename, lines=True)
    if not is_ref:
        assert ('hard_labels' in df.columns) or ('soft_labels' in df.columns), \
            f'File {filename} contains no predicted label!'
        if 'hard_labels' not in df.columns:
            df['hard_labels'] = df.soft_labels.apply(recompute_hard_labels)
        elif 'soft_labels' not in df.columns:
            df['soft_labels'] = df.hard_labels.apply(infer_soft_labels)
    # adding an extra column for convenience
    columns = ['id', 'soft_labels', 'hard_labels']
    if is_ref:
        df['text_len'] = df.model_output_text.apply(len)
        columns += ['text_len']
    df = df[columns]
    return df.sort_values('id').to_dict(orient='records')

def score_iou(ref_dict, pred_dict):
    """computes intersection-over-union between reference and predicted hard labels, for a single datapoint.
    inputs:
    - ref_dict: a gold reference datapoint,
    - pred_dict: a model's prediction
    returns:
    the IoU, or 1.0 if neither the reference nor the prediction contain hallucinations
    """
    # ensure the prediction is correctly matched to its reference
    assert ref_dict['id'] == pred_dict['id']
    # convert annotations to sets of indices
    ref_indices = {idx for span in ref_dict['hard_labels'] for idx in range(*span)}
    pred_indices = {idx for span in pred_dict['hard_labels'] for idx in range(*span)}
    # avoid division by zero
    if not pred_indices and not ref_indices: return 1.
    # otherwise compute & return IoU
    return len(ref_indices & pred_indices) / len(ref_indices | pred_indices)

def score_cor(ref_dict, pred_dict):
    """computes Spearman correlation between predicted and reference soft labels, for a single datapoint.
    inputs:
    - ref_dict: a gold reference datapoint,
    - pred_dict: a model's prediction
    returns:
    the Spearman correlation, or a binarized exact match (0.0 or 1.0) if the reference or prediction contains no variation
    """
    # ensure the prediction is correctly matched to its reference
    assert ref_dict['id'] == pred_dict['id']
    # convert annotations to vectors of observations
    ref_vec = [0.] * ref_dict['text_len']
    pred_vec = [0.] * ref_dict['text_len']
    for span in ref_dict['soft_labels']:
        for idx in range(span['start'], span['end']):
            ref_vec[idx] = span['prob']
    for span in pred_dict['soft_labels']:
        for idx in range(span['start'], span['end']):
            pred_vec[idx] = span['prob']
    # constant series (i.e., no hallucination) => cor is undef
    if len({round(flt, 8) for flt in pred_vec}) == 1 or len({round(flt, 8) for flt in ref_vec}) == 1 : 
        return float(len({round(flt, 8) for flt in ref_vec}) == len({round(flt, 8) for flt in pred_vec}))
    # otherwise compute Spearman's rho
    return spearmanr(ref_vec, pred_vec).correlation

def get_scores(prompt, ref_dicts, pred_dicts, comment, output_file=None, temp=-1):
    assert len(ref_dicts) == len(pred_dicts)
    
    # Рассчитываем метрики
    ious = np.array([score_iou(r, d) for r, d in zip(ref_dicts, pred_dicts)])
    cors = np.array([score_cor(r, d) for r, d in zip(ref_dicts, pred_dicts)])
    
    # Формируем данные для записи
    metrics = {
        "iou": ious.mean(),
        "corr": cors.mean(),
        "prompt": prompt,
        "temp": temp,
        "time": datetime.now().isoformat(),  # Текущее время в ISO формате
        "comment": comment
    }
    
    if output_file is not None:
        try:
            # Читаем существующий JSON-файл, если он есть
            with open(output_file, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
        except FileNotFoundError:
            # Если файла нет, создаём новый словарь
            data = {}
        
        # Определяем новый уникальный идентификатор (ключ)
        new_id = str(len(data))
        data[new_id] = metrics
        
        # Записываем обновлённые данные в файл
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
    
    return ious, cors

def get_scores_refine(df, prompt, refine_prompt, ref_dicts, pred_dicts, comment, output_file=None, temp=-1, refine_type='score'):
    assert len(ref_dicts) == len(pred_dicts)
    
    # Рассчитываем метрики
    ious = np.array([score_iou(r, d) for r, d in zip(ref_dicts, pred_dicts)])
    cors = np.array([score_cor(r, d) for r, d in zip(ref_dicts, pred_dicts)])
    
    # Формируем данные для записи
    metrics = {
        "iou": ious.mean(),
        "corr": cors.mean(),
        "prompt": prompt,
        "refine_type": refine_type,
        "refine_prompt": refine_prompt,
        "temp": temp,
        "time": datetime.now().isoformat(),  # Текущее время в ISO формате
        "comment": comment
    }
    
    if output_file is not None:
        try:
            # Читаем существующий JSON-файл, если он есть
            with open(output_file, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
        except FileNotFoundError:
            # Если файла нет, создаём новый словарь
            data = {}
        
        # Определяем новый уникальный идентификатор (ключ)
        new_id = str(len(data))
        data[new_id] = metrics
        df.to_json(f'/data/levykin/semeval2025_levykin/hallucinations_with_marks/detection_with_refine_exp_{new_id}.json', orient='records')
        
        # Записываем обновлённые данные в файл
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
    
    return ious, cors

if __name__ == '__main__':
    p = ap.ArgumentParser()
    p.add_argument('--ref-file', type=str)
    p.add_argument('--pred-file', type=str)
    p.add_argument('--output-file', type=str)
    p.add_argument('--comment', type=str)

    a = p.parse_args()
    _ = get_scores(
        ref_dicts=load_jsonl_file_to_records(a.ref_file), 
        pred_dicts=load_jsonl_file_to_records(a.pred_file, is_ref=False), 
        output_file=a.output_file, 
        comment=a.comment
    )

