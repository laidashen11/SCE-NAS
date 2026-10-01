
import os
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from genotypesNAS201 import Structure
from openai import OpenAI

LLM_MODEL = "deepseek-v4-flash"
API_BASE_URL = "https://api.deepseek.com"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 49152
LLM_TIMEOUT = 150
LLM_MAX_RETRIES = 3

def alphas_to_genotype(alphas_normal, op_names, max_nodes=4):
    
    alphas_softmax = F.softmax(alphas_normal, dim=-1)

    edge2index = {}
    idx = 0
    for i in range(1, max_nodes):          
        for j in range(i):                 
            edge2index["{}<-{}".format(i, j)] = idx
            idx += 1

    genotypes = []
    for i in range(1, max_nodes):
        xlist = []
        for j in range(i):
            edge_idx = edge2index["{}<-{}".format(i, j)]
            op_idx = alphas_softmax[edge_idx].argmax().item()
            xlist.append((op_names[op_idx], j))
        genotypes.append(tuple(xlist))

    return Structure(genotypes)

def alphas_to_onehot(alphas_normal, op_names, max_nodes=4):
    
    alphas_softmax = F.softmax(alphas_normal, dim=-1)
    indices = alphas_softmax.argmax(dim=-1).cpu()          
    num_edges, num_ops = indices.shape[0], len(op_names)
    onehot = torch.zeros(num_edges, num_ops)
    for i in range(num_edges):
        onehot[i, indices[i]] = 1.0
    return onehot.flatten().numpy()                        

def alphas_to_genotype_vec(alphas_normal, op_names, max_nodes=4):
    
    alphas_softmax = F.softmax(alphas_normal, dim=-1)
    return alphas_softmax.argmax(dim=-1).cpu().numpy()     

def hamming_distance(v1, v2):
    
    return int(np.count_nonzero(v1 != v2))

def query_nasbench_true_acc(api, structure, dataset="cifar10"):
    
    arch_index = api.query_index_by_arch(structure)
    info = api.query_by_index(arch_index, hp="200")
    if dataset == "cifar10":
        valid_acc = info.get_metrics("cifar10-valid", "x-valid")["accuracy"] / 100.0
        test_acc = info.get_metrics("cifar10", "ori-test")["accuracy"] / 100.0
    else:
        valid_acc = info.get_metrics(dataset, "x-valid")["accuracy"] / 100.0
        test_acc = info.get_metrics(dataset, "x-test")["accuracy"] / 100.0
    return valid_acc, test_acc

def _render_standard_examples(pkb_examples):
    lines = []
    for ex in pkb_examples:
        lines.append(
            "    <Architecture Information>: {}; <Value>: {:.6f};".format(
                ex["genotype_str"], ex["true_acc_valid"])
        )
    return "\n".join(lines)

NB201_DATASET_INFO = {
    "cifar10": {   
        "display_name": "CIFAR-10",
        "acc_low": 0.097120,
        "acc_high": 0.916067,
        "optimal_arch": "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|",
        "layers": [
            (lambda a: 0.85 <= a < 0.88, 4),
            (lambda a: 0.88 <= a < 0.90, 5),
            (lambda a: a >= 0.90, 5),
        ],
    },
    "cifar100": {  
        "display_name": "CIFAR-100",
        "acc_low": 0.010000,
        "acc_high": 0.734933,
        "optimal_arch": "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|",
        "layers": [
            (lambda a: 0.66 <= a < 0.69, 4),
            (lambda a: 0.69 <= a < 0.71, 5),
            (lambda a: a >= 0.71, 5),
        ],
    },
    "ImageNet16-120": {  
        "display_name": "ImageNet16-120",
        "acc_low": 0.008333,
        "acc_high": 0.467667,
        "optimal_arch": "|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|",
        "layers": [
            (lambda a: 0.40 <= a < 0.43, 4),
            (lambda a: 0.43 <= a < 0.45, 5),
            (lambda a: a >= 0.45, 5),
        ],
    },
}

def build_dynamic_examples(static_anchors, pool, total=20, dataset="cifar10", layers=None, fill_random=False):
    
    if layers is None:
        layers = NB201_DATASET_INFO[dataset]["layers"]
    picked = list(static_anchors)
    for cond, quota in layers:
        cands = [e for e in pool if cond(e["true_acc_valid"])]
        np.random.shuffle(cands)
        picked.extend(cands[:quota])
    if fill_random and len(picked) < total:
        have = {e["genotype_str"] for e in picked}
        extra = [e for e in pool if e["genotype_str"] not in have]
        np.random.shuffle(extra)
        picked.extend(extra[:total - len(picked)])
    picked.sort(key = lambda e: e["true_acc_valid"])
    return picked[:total]

def _render_emb_entries(emb_entries):
    
    lines = []
    for ex in emb_entries:
        lines.append(
            "    <Architecture Information>: {}; "
            "<Predicted Accuracy>: {:.6f}; "
            "<True Accuracy>: {:.6f}; "
            "<Prediction Error>: {:.6f};".format(
                ex["genotype_str"], ex["pred_acc"], ex["true_acc"], ex["error"])
        )
    return "\n".join(lines)

def build_emb_prompt(pkb_examples, emb_entries, genotype_str, dataset="cifar10"):
    
    pkb_text = _render_standard_examples(pkb_examples)
    emb_text = _render_emb_entries(emb_entries)
    dinfo = NB201_DATASET_INFO[dataset]
    ds_display = dinfo["display_name"]
    ds_low = "{:.6f}".format(dinfo["acc_low"])
    ds_high = "{:.6f}".format(dinfo["acc_high"])
    range_line = "- {} validation accuracy range: {} (worst) ~ {} (best)".format(ds_display, ds_low, ds_high)

    prompt = f"""You are an expert Accuracy Predictor for CNN Architectures (NAS-Bench-201 Space).

### Task:
You need to predict the validation accuracy (Value) for a given Candidate Architecture based on a example architecture collection. This task involves inferring the behavior of a black-box function and understanding the non-linear relationships between architecture attributes and their corresponding test accuracy Value.

### Search Space:
- NAS-Bench-201 has 4 nodes (0=input, 1/2/3=intermediate), 6 directed edges with fixed topology:
  Node1<-0, Node2<-0/1, Node3<-0/1/2
- Genotype format: |op1~0|+|op2~0|op3~1|+|op4~0|op5~1|op6~2|
- Each edge selects one operation from: none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3
{range_line}

### Example architecture collection ({len(pkb_examples)} sample architectures, sorted in ascending order of accuracy.):
{pkb_text}

### Error Memory Bank (architectures you previously predicted with large errors):
{emb_text}

### Steps:
1. Analyze all samples in the example architecture collection and historical error set to summarize the hidden mapping rule between architecture feature vector and real performance Value.
2. Based on the mapping relationships obtained in the previous steps, the candidate architecture is compared with the similar architectures in the example architecture set and the historical error set.
3. Finally, predict the approximate Value for the Candidate Architecture below.

### Candidate Architecture (predict this):
<Architecture Information>: {genotype_str}; <Value>: ;

### Constraints:
1. The output must be in the following fixed and valid JSON format, without any additional text.
2. The range of the output accuracy rate is: {ds_low} <= Predicted Accuracy <= {ds_high}.
3. The precision is rounded to a maximum of six decimal places.

### OUTPUT FORMAT:
{{"predicted_accuracy": 0.xxxxx}}"""
    return prompt

def parse_prediction_response(response_text):
    
    if not response_text:
        return None

    response_text = response_text.strip()
    if response_text.startswith("```"):
        first_newline = response_text.find("\n")
        if first_newline != -1:
            response_text = response_text[first_newline:]
        end_marker = response_text.rfind("```")
        if end_marker != -1:
            response_text = response_text[:end_marker]
        response_text = response_text.strip()

    try:
        start = response_text.index("{")
        end = response_text.rindex("}") + 1
        data = json.loads(response_text[start:end])
        for key in ["predicted_accuracy", "Value", "accuracy"]:
            if key in data:
                acc = float(data[key])
                if 0.0 <= acc <= 1.0:
                    return acc
    except (ValueError, json.JSONDecodeError):
        pass

    import re
    numbers = re.findall(r"0\.\d{4,}", response_text)
    if numbers:
        acc = float(numbers[0])
        if 0.0 <= acc <= 1.0:
            return acc
    return None

def create_client(api_key):
    
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)

def call_llm(client, prompt, max_retries=LLM_MAX_RETRIES):
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert Accuracy Predictor."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                reasoning_effort="medium",
                extra_body={"thinking": {"type": "disabled"}},
                timeout=LLM_TIMEOUT,
            )
            llm_text = response.choices[0].message.content
            acc = parse_prediction_response(llm_text)
            if acc is not None:
                return acc, True
            print("  [Warning] Attempt {}: could not parse".format(attempt + 1))
        except Exception as e:
            print("  [Warning] Attempt {}: API error: {}".format(attempt + 1, e))
            if attempt < max_retries - 1:
                time.sleep(3)
    return None, False

def build_classify_prompt(pkb_examples, gstr_parent, gstr_child, dataset="cifar10"):
    
    pkb_text = _render_standard_examples(pkb_examples)
    dinfo = NB201_DATASET_INFO[dataset]
    ds_display = dinfo["display_name"]
    ds_low = "{:.6f}".format(dinfo["acc_low"])
    ds_high = "{:.6f}".format(dinfo["acc_high"])
    ds_optimal_arch = dinfo["optimal_arch"]
    range_line = "- {} validation accuracy range: {} (worst) ~ {} (best)".format(ds_display, ds_low, ds_high)

    prompt = f"""You are an expert Accuracy Ranker for CNN Architectures (NAS-Bench-201 Space).

### Task:
Given a parent architecture and its mutated child architectures, please determine which architecture achieves a higher accuracy on the {ds_display} validation set, based on the provided set of example architectures and the effects of different operations.

### Search Space:
- NAS-Bench-201 has 4 nodes (0=input, 1/2/3=intermediate), 6 directed edges with fixed topology:
  Node1<-0, Node2<-0/1, Node3<-0/1/2
- Genotype format: |op1~0|+|op2~0|op3~1|+|op4~0|op5~1|op6~2|
- Each edge selects one operation from: none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3
{range_line}

### Example architecture collection ({len(pkb_examples)} sample architectures, sorted in ascending order of accuracy.):
{pkb_text}

### Operation Analysis (how each operation affects performance):
- `none` — Cuts the connection. Makes the network shallower. Useful only when the edge is redundant. **Effect: reduces capacity, likely harmful.**
- `skip_connect` — Identity skip connection. Helps gradient flow during training but does not extract features. Adds minimal parameters. **Effect: moderate, better than none but weaker than convolutions.**
- `nor_conv_1x1` — 1x1 convolution. Learns channel-wise transformations with few parameters. Good for cross-channel information mixing. **Effect: moderate capacity, useful but limited expressiveness.**
- `nor_conv_3x3` — 3x3 convolution. Strongest feature extractor in this search space. Learns spatial patterns effectively. **Effect: highest capacity, most beneficial for accuracy.**
- `avg_pool_3x3` — 3x3 average pooling. Smooths features, reduces spatial dimension information. Very few parameters. **Effect: limited learning capacity, weaker than convolutions.**

### Steps:
1. Analyze all samples in the example architecture collection to summarize the hidden mapping rule between architecture feature and real performance Value.
2. Based on this mapping rule and the possible outcomes of different operations, compare the parent and child architectures below.
3. Classify which one is more likely to have HIGHER validation accuracy.

### Candidate Architectures (compare):
<Parent Architecture>: {gstr_parent}
<Child Architecture>: {gstr_child}

### Constraints:
1. The output must be a valid JSON object with a single key "choice", without any additional text.
2. "choice": 0 means the Parent is better; "choice": 1 means the Child is better.
3. Do NOT output any numeric accuracy values.

### OUTPUT FORMAT:
{{"choice": 0}}"""
    return prompt

def parse_classify_response(response_text):
    
    if not response_text:
        return None

    response_text = response_text.strip()
    if response_text.startswith("```"):
        first_newline = response_text.find("\n")
        if first_newline != -1:
            response_text = response_text[first_newline:]
        end_marker = response_text.rfind("```")
        if end_marker != -1:
            response_text = response_text[:end_marker]
        response_text = response_text.strip()

    try:
        start = response_text.index("{")
        end = response_text.rindex("}") + 1
        data = json.loads(response_text[start:end])
        if "choice" in data:
            choice = int(data["choice"])
            if choice in (0, 1):
                return choice
    except (ValueError, json.JSONDecodeError, KeyError):
        pass
    return None

def llm_classify_pair(client, pkb_examples, gstr_parent, gstr_child, max_retries=LLM_MAX_RETRIES, dataset="cifar10"):
    
    prompt = build_classify_prompt(pkb_examples, gstr_parent, gstr_child, dataset=dataset)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert Accuracy Ranker."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                reasoning_effort="medium",
                extra_body={"thinking": {"type": "enabled"}},
                timeout=LLM_TIMEOUT,
            )
            llm_text = response.choices[0].message.content
            choice = parse_classify_response(llm_text)
            if choice is not None:
                return choice, True
            print("  [Warning] Classify attempt {}: could not parse".format(attempt + 1))
        except Exception as e:
            print("  [Warning] Classify attempt {}: API error: {}".format(attempt + 1, e))
            if attempt < max_retries - 1:
                time.sleep(3)
    return None, False

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

def compute_values(entries):
    
    n = len(entries)
    if n == 0:
        return entries
    errors = np.array([e["error"] for e in entries], dtype=float)
    max_err = errors.max()
    e_norm = errors / max_err if max_err > 0 else np.zeros(n)

    if n == 1:
        max_sim = np.zeros(n)
    else:
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim[i, j] = cosine_similarity(entries[i]["arch_vector"], entries[j]["arch_vector"])
        max_sim = np.clip(sim.max(axis=1), 0.0, 1.0)

    out = []
    for i, ent in enumerate(entries):
        out.append({
            **ent,
            "value": 0.5 * float(e_norm[i]) + 0.5 * (1.0 - float(max_sim[i])),
        })
    return out

def find_match(emb, entry, threshold=0.95):
    
    if not emb:
        return None
    best_idx, best_sim = None, -1.0
    for i, e in enumerate(emb):
        s = cosine_similarity(entry["arch_vector"], e["arch_vector"])
        if s > best_sim:
            best_idx, best_sim = i, s
    return best_idx if best_sim > threshold else None

def update_emb(emb, new_entries, cap=20, threshold=0.95):
    
    emb = [dict(e) for e in emb]
    for entry in new_entries:
        match_idx = find_match(emb, entry, threshold)
        if match_idx is not None:
            emb[match_idx] = dict(entry)          
        else:
            emb.append(dict(entry))               

    pool = compute_values(emb)
    pool.sort(key=lambda e: e["value"], reverse=True)
    return pool[:cap]

def emb_max_sim(emb, arch_vector):
    
    if not emb:
        return 0.0
    return max(cosine_similarity(arch_vector, e["arch_vector"]) for e in emb)

def emb_confidence(emb, arch_vector, p=2.0, e_ref=0.05):
    
    if not emb:
        return 1.0
    w = np.asarray([max(cosine_similarity(arch_vector, e["arch_vector"]), 0.0) ** p for e in emb], dtype=float)
    if w.sum() <= 0:
        return 1.0
    errs = np.asarray([float(e.get("error", 0.0)) for e in emb], dtype=float)
    local_err = float((w * errs).sum() / w.sum())
    return float(np.clip(1.0 - local_err / e_ref, 0.0, 1.0))

def select_diverse_candidates(top_idx, all_idx, genotype_vecs, d):
    
    top_set = set(top_idx)
    others = [i for i in all_idx if i not in top_set]
    if d <= 0 or not others or not top_idx:
        return []
    scored = []
    for i in others:
        min_dist = min(hamming_distance(genotype_vecs[i], genotype_vecs[t]) for t in top_idx)
        scored.append((min_dist, i))
    scored.sort(key = lambda x: x[0], reverse = True)
    return [i for _, i in scored[:d]]

def rank_fusion(cand, arch_vecs, emb, llm_pred, no_llm, sup_rank = None, e_ref = 0.05):
    
    k = len(cand)
    if sup_rank is None:
        sup_rank = {idx: pos + 1 for pos, idx in enumerate(cand)}
    denom = max(sup_rank.values()) if sup_rank else k     

    valid = [i for i in cand if (not no_llm) and llm_pred.get(i) is not None]
    valid_sorted = sorted(valid, key = lambda i: llm_pred[i], reverse = True)
    llm_rank = {idx: pos + 1 for pos, idx in enumerate(valid_sorted)}

    finals = {}
    for idx in cand:
        if (not no_llm) and llm_pred.get(idx) is not None:
            conf = emb_confidence(emb, arch_vecs[idx], e_ref = e_ref)
        else:
            conf = 0.0                                       
        rank_super = (denom + 1 - sup_rank[idx]) / denom     
        rank_llm = (k + 1 - llm_rank.get(idx, sup_rank[idx])) / k   
        finals[idx] = conf * rank_llm + (1.0 - conf) * rank_super
    return finals

def greedy_truncate(order, genotype_vecs, k, theta):
    
    selected, selected_set = [], set()
    for idx in order:
        if len(selected) >= k:
            break
        v = genotype_vecs[idx]
        if all(hamming_distance(v, genotype_vecs[j]) >= theta for j in selected):
            selected.append(idx)
            selected_set.add(idx)

    for idx in order:
        if len(selected) >= k:
            break
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
    return selected

NB101_DATASET_INFO = {
    "cifar10": {   
        "display_name": "CIFAR-10",
        "acc_low": 0.095052,      
        "acc_high": 0.945913,     
        "layers": [
            (lambda a: 0.920 <= a < 0.932, 5),
            (lambda a: 0.932 <= a < 0.940, 5),
            (lambda a: a >= 0.940, 4),
        ],
    },
}

def alphas_to_onehot101(alphas_ops, alphas_edges, threshold=0.5):
    
    ops_sm = F.softmax(alphas_ops, dim=-1)
    ops_idx = ops_sm.argmax(dim=-1).cpu()
    onehot_ops = torch.zeros(alphas_ops.shape[0], alphas_ops.shape[1])
    for i in range(ops_idx.shape[0]):
        onehot_ops[i, ops_idx[i]] = 1.0
    edge_bits = (alphas_edges > threshold).view(-1).cpu().numpy().astype(float)
    return np.concatenate([onehot_ops.flatten().numpy(), edge_bits])

def query_nasbench_true_acc101(api, structure, mode="mean"):
    
    from nasbench.lib.model_spec import ModelSpec
    spec = ModelSpec(matrix=structure.adj_matrix,
                     ops=['input'] + list(structure.ops) + ['output'])
    if mode == "single":
        metrics = api.query(spec, epochs=108)
        return float(metrics["validation_accuracy"]), float(metrics["test_accuracy"])
    _, computed = api.get_metrics_from_spec(spec)
    dpoints = computed[108]
    valid_acc = float(np.mean([d['final_validation_accuracy'] for d in dpoints]))
    test_acc = float(np.mean([d['final_test_accuracy'] for d in dpoints]))
    return valid_acc, test_acc

_NB101_SPACE = """- NAS-Bench-101 has 7 nodes in a cell: node 0 = input, nodes 1-5 = 5 intermediate nodes, node 6 = output.
- Each intermediate node selects one of 3 operations: conv3x3-bn-relu, conv1x1-bn-relu, maxpool3x3.
- Edges form a DAG: input(0)->intermediate(1-5), intermediate->intermediate, intermediate->output(6).
- Genotype format: |op1|op2|op3|op4|op5|+|src->dst;src->dst;...|  (5 operations, then a list of directed edges).
- CIFAR-10 test accuracy of good architectures is around 0.92 ~ 0.95."""

def build_emb_prompt101(pkb_examples, emb_entries, genotype_str):
    
    pkb_text = _render_standard_examples(pkb_examples)
    emb_text = _render_emb_entries(emb_entries)

    prompt = f"""You are an expert Accuracy Predictor for CNN Architectures (NAS-Bench-101 Space).

### Task:
You need to predict the validation accuracy (Value) for a given Candidate Architecture based on an example architecture collection. This task involves inferring the behavior of a black-box function and understanding the non-linear relationships between architecture attributes and their corresponding accuracy Value.

### Search Space:
{_NB101_SPACE}

### Example architecture collection ({len(pkb_examples)} sample architectures, sorted in ascending order of accuracy.):
{pkb_text}

### Error Memory Bank (architectures you previously predicted with large errors):
{emb_text}

### Steps:
1. Analyze all samples in the example architecture collection and historical error set to summarize the hidden mapping rule between architecture feature vector and real performance Value.
2. Based on the mapping relationships obtained in the previous steps, the candidate architecture is compared with the similar architectures in the example architecture set and the historical error set.
3. Finally, predict the approximate Value for the Candidate Architecture below.

### Candidate Architecture (predict this):
<Architecture Information>: {genotype_str}; <Value>: ;

### Constraints:
1. The output must be in the following fixed and valid JSON format, without any additional text.
2. The range of the output accuracy rate is: 0.90 <= Predicted Accuracy <= 0.95.
3. The precision is rounded to a maximum of six decimal places.

### OUTPUT FORMAT:
{{"predicted_accuracy": 0.xxxxx}}"""
    return prompt

def build_classify_prompt101(pkb_examples, gstr_parent, gstr_child):
    
    pkb_text = _render_standard_examples(pkb_examples)

    prompt = f"""You are an expert Accuracy Ranker for CNN Architectures (NAS-Bench-101 Space).

### Task:
Given a parent architecture and its mutated child architecture, classify which one has higher CIFAR-10 validation accuracy. Do NOT output numeric scores - only decide the relative order.

### Search Space:
{_NB101_SPACE}

### Example architecture collection ({len(pkb_examples)} sample architectures, sorted in ascending order of accuracy.):
{pkb_text}

### Steps:
1. Analyze all samples in the example architecture collection to summarize the hidden mapping rule between architecture feature and real performance Value.
2. Compare the Parent and Child architectures below using that mapping rule.
3. Classify which one is more likely to have HIGHER validation accuracy.

### Candidate Architectures (compare):
<Parent Architecture>: {gstr_parent}
<Child Architecture>: {gstr_child}

### Constraints:
1. The output must be a valid JSON object with a single key "choice", without any additional text.
2. "choice": 0 means the Parent is better; "choice": 1 means the Child is better.
3. Do NOT output any numeric accuracy values.

### OUTPUT FORMAT:
{{"choice": 0}}"""
    return prompt

def llm_classify_pair101(client, pkb_examples, gstr_parent, gstr_child, max_retries=LLM_MAX_RETRIES):
    
    prompt = build_classify_prompt101(pkb_examples, gstr_parent, gstr_child)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert Accuracy Ranker."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                reasoning_effort="medium",
                extra_body={"thinking": {"type": "enabled"}},
                timeout=LLM_TIMEOUT,
            )
            llm_text = response.choices[0].message.content
            choice = parse_classify_response(llm_text)
            if choice is not None:
                return choice, True
            print("  [Warning] Classify(101) attempt {}: could not parse".format(attempt + 1))
        except Exception as e:
            print("  [Warning] Classify(101) attempt {}: API error: {}".format(attempt + 1, e))
            if attempt < max_retries - 1:
                time.sleep(3)
    return None, False
