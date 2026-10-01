
import os
import sys
import logging
import random
import torch.nn as nn
import argparse
import numpy as np
import pickle
import json
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import utils

from cell_operationsNAS101 import NAS101SearchCell
from genotypesNAS101 import (NAS_BENCH_101_OPS, alphas_to_structure,
                             NUM_INTERMEDIATE, NUM_OPS, NUM_EDGES)
from deNAS101                import DifferentialEvolution
from emb_llm                 import (alphas_to_onehot101, query_nasbench_true_acc101,
                                     build_emb_prompt101, call_llm,
                                     create_client, update_emb, rank_fusion, llm_classify_pair101,
                                     emb_max_sim, build_dynamic_examples, NB101_DATASET_INFO)

from populationNAS101        import *
from optimizers              import get_optim_scheduler
from search_model_NAS101     import TinyNetwork
from torch.utils.tensorboard import SummaryWriter

DEEPSEEK_API_KEY = ""  # TODO: set your DeepSeek API key (or set env DEEPSEEK_API_KEY)
NASBENCH101_PATH = "./nasbench_only108.tfrecord"  # TODO: set the path to the NAS-Bench-101 file

parser = argparse.ArgumentParser("NAS101")
parser.add_argument('--data', type = str, default = '', help = 'location of the data corpus (set this to your dataset root)')
parser.add_argument('--dir', type = str, default = None, help = 'location of trials')
parser.add_argument('--cutout', action = 'store_true', default = False, help = 'use cutout')
parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
parser.add_argument('--valid_batch_size', type = int, default = 1024, help = 'validation batch size')
parser.add_argument('--epochs', type = int, default = 50, help = 'num of generations (cosine LR schedule auto-adapts)')
parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
parser.add_argument('--gpu', type = int, default = 0, help = 'gpu device id')
parser.add_argument('--tsize', type = int, default = 10, help = 'Tournament size (unused in DE, kept for interface compatibility)')
parser.add_argument('--mutate_rate', type = float, default = 0.1, help = 'mutation rate (unused in DE, kept for interface compatibility)')
parser.add_argument('--learning_rate', type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min', type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
parser.add_argument('--weight_decay', type = float, default = 3e-4, help = 'weight decay')
parser.add_argument('--grad_clip', type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--pop_size', type = int, default = 50, help = 'population size')
parser.add_argument('--init_channels', type = int, default = 16, help = 'num of init channels')
parser.add_argument('--num_cells', type = int, default = 3, help = 'number of cells per stack (NB101 uses 3 cells per layer)')
parser.add_argument('--track_running_stats', action = 'store_true', default = False, help = 'use track_running_stats in BN layer')
parser.add_argument('--api_path', type = str, default = NASBENCH101_PATH, help = 'NAS-Bench-101 tfrecord path')
parser.add_argument('--workers', type = int, default = 2, help = 'number of data loading workers (default: 2)')
parser.add_argument('--config_path', type = str, default = './configs/CIFAR.config', help = 'The config path.')

parser.add_argument('--llm_api_key', type = str, default = DEEPSEEK_API_KEY, help = 'DeepSeek API key (default: DEEPSEEK_API_KEY constant)')
parser.add_argument('--no_llm', action = 'store_true', default = False, help = 'skip LLM calls (use supernet score as placeholder)')
parser.add_argument('--pair_topk', type = int, default = 20, help = 'number of supernet-top archs whose parent-child pairs get LLM classification (default 20; regression covers the <=pair_topk winners too)')
parser.add_argument('--llm_topk', type = int, default = 20, help = 'LLM regression upper bound (gen>=2: marked-pair winners; gen1: supernet top-k)')
parser.add_argument('--query_topk', type = int, default = 8, help = 'number of archs queried on NAS-Bench-101 per generation (total = query_topk * epochs)')
parser.add_argument('--emb_cap', type = int, default = 20, help = 'Error Memory Bank capacity')
parser.add_argument('--sim_threshold', type = float, default = 0.8, help = 'EMB dedup similarity threshold')
parser.add_argument('--query_mode', type = str, default = 'mean', choices = ['mean', 'single'],
                    help = 'NB101 ground-truth query mode: mean=average over repeats (default, deterministic); single=random single repeat via official api.query (replicates old experiments)')
parser.add_argument('--edge_relax', type = str, default = 'sparse', choices = ['sparse', 'dense'],
                    help = 'NB101 edge relaxation: sparse=only <=9 discrete edges (mask, consistent with ground truth, default); dense=legacy sigmoid over all 20 edges')
parser.add_argument('--fewshot_path', type = str, default = './fewshot_archs101.json', help = 'few-shot examples file (NB101)')
parser.add_argument('--no_debug_print', action = 'store_true', default = False, help = 'disable debug truth printing (default: ON)')
args = parser.parse_args()

def train(model, train_queue, criterion, optimizer, gen):
    model.train()
    for step, (inputs, targets) in enumerate(train_queue):
        p = population.get_population()[step % args.pop_size]
        model.update_alphas(p.arch_parameters[0], p.arch_parameters[1])
        model.discretize()   

        n = inputs.size(0)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        _, logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        p.objs.update(loss.data.cpu().item(), n)
        p.top1.update(prec1.data.cpu().item(), n)
        p.top5.update(prec5.data.cpu().item(), n)

        if (step + 1) % 100 == 0:
            logging.info("[{} Generation] Training batch #{}, {}/{} arch: loss {:.4f}, prec1 {:.2f}, prec5 {:.2f}".format(
                gen, step, step % args.pop_size, len(population.get_population()),
                p.objs.avg, p.top1.avg, p.top5.avg))

def validation(model, valid_queue, criterion, gen):
    model.eval()
    for i in range(len(population.get_population())):
        p = population.get_population()[i]
        model.update_alphas(p.arch_parameters[0], p.arch_parameters[1])
        model.discretize()
        p.objs.reset()
        p.top1.reset()
        p.top5.reset()
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(valid_queue):
                n = inputs.size(0)
                inputs = inputs.to(device)
                targets = targets.to(device)
                _, logits = model(inputs)
                loss = criterion(logits, targets)
                prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                p.objs.update(loss.data.cpu().item(), n)
                p.top1.update(prec1.data.cpu().item(), n)
                p.top5.update(prec5.data.cpu().item(), n)

    global emb, QUERY_COUNT, LLM_CALL_COUNT, hall_of_fame, truth_pool, best_by_test

    if gen >= 10 and truth_pool:
        best_static = [max(fewshot_examples, key = lambda e: e["true_acc_valid"])]
        fixed_anchors = fewshot_examples[:5] + best_static
        examples = build_dynamic_examples(fixed_anchors, truth_pool,
                                          layers = NB101_DATASET_INFO["cifar10"]["layers"],
                                          fill_random = True)
        if gen % 5 == 0:
            with open(os.path.join(DIR, "fewshot_epoch_{}.json".format(gen)), 'w', encoding = 'utf-8') as f:
                json.dump(examples, f, indent = 2, ensure_ascii = False)
    else:
        examples = fewshot_examples

    pop = population.get_population()
    size = len(pop)
    sup_scores = np.array([p.top1.avg for p in pop], dtype=float)
    arch_vecs = {i: alphas_to_onehot101(p.arch_parameters[0], p.arch_parameters[1]) for i, p in enumerate(pop)}

    marked_winners_new = None

    if size > args.pop_size:
        sup_order = list(np.argsort(-sup_scores))
        marked = set(sup_order[:args.pair_topk])
        marked_pairs = sorted({m if m < args.pop_size else m - args.pop_size for m in marked})

        if not args.no_debug_print:
            logging.info("[DEBUG] === {} archs sorted by supernet (before 1v1 selection) ===".format(size))
            best_row_100 = None
            for pos, idx in enumerate(sup_order, start=1):
                struct_tmp = alphas_to_structure(pop[idx].arch_parameters[0], pop[idx].arch_parameters[1])
                gstr = struct_tmp.tostr()
                v_acc, t_acc = query_nasbench_true_acc101(api, struct_tmp, mode = args.query_mode)
                emb_cos = emb_max_sim(emb, arch_vecs[idx])
                emb_cos_s = "{:.4f}".format(emb_cos) if emb else "----"
                ptype = "P" if idx < args.pop_size else "C"
                logging.info("[DEBUG] {:3d} | idx={:3d} | type={} | supernet={:8.4f} | truth_valid={:.6f} | truth_test={:.6f} | emb_cos={:>6} | {}".format(
                    pos, idx, ptype, sup_scores[idx], v_acc, t_acc, emb_cos_s, gstr))
                if best_row_100 is None or v_acc > best_row_100[2]:
                    best_row_100 = (pos, gstr, v_acc, t_acc)
            if best_row_100 is not None:
                logging.info("[DEBUG] BEST among these {} (by truth): pos={} | true_valid={:.6f} | test={:.6f} | {}".format(
                    size, best_row_100[0], best_row_100[2], best_row_100[3], best_row_100[1]))

        drop = []
        marked_winners_orig = []
        for i in range(args.pop_size):
            p_idx, c_idx = i, i + args.pop_size
            if i in marked_pairs and not args.no_llm:
                gstr_p = alphas_to_structure(pop[p_idx].arch_parameters[0], pop[p_idx].arch_parameters[1]).tostr()
                gstr_c = alphas_to_structure(pop[c_idx].arch_parameters[0], pop[c_idx].arch_parameters[1]).tostr()
                choice, ok = llm_classify_pair101(client, examples, gstr_p, gstr_c)
                LLM_CALL_COUNT += 1
                keep_parent = (choice == 0) if ok else (sup_scores[p_idx] > sup_scores[c_idx])
            else:
                keep_parent = sup_scores[p_idx] > sup_scores[c_idx]
            drop.append(c_idx if keep_parent else p_idx)
            if i in marked_pairs:
                marked_winners_orig.append(p_idx if keep_parent else c_idx)

        drop_set = set(drop)
        kept_orig = [k for k in range(size) if k not in drop_set]
        orig_to_new = {orig: pos for pos, orig in enumerate(kept_orig)}
        marked_winners_new = [orig_to_new[o] for o in marked_winners_orig]
        population.pop_pop(drop)
        pop = population.get_population()
        size = len(pop)
        sup_scores = np.array([p.top1.avg for p in pop], dtype=float)
        arch_vecs = {i: alphas_to_onehot101(p.arch_parameters[0], p.arch_parameters[1]) for i, p in enumerate(pop)}
        logging.info("[INFO] Generation {}: 1v1 selection -> {} archs ({} pairs LLM-classified)".format(
            gen, size, len(marked_pairs)))

    sup_order = list(np.argsort(-sup_scores))
    sup_rank = {idx: pos + 1 for pos, idx in enumerate(sup_order)}
    cand = marked_winners_new if marked_winners_new is not None else sup_order[:args.llm_topk]

    llm_pred = {}
    for idx in cand:
        gstr = alphas_to_structure(pop[idx].arch_parameters[0], pop[idx].arch_parameters[1]).tostr()
        if not args.no_llm:
            prompt = build_emb_prompt101(examples, emb, gstr)
            pred, ok = call_llm(client, prompt)
            LLM_CALL_COUNT += 1
            llm_pred[idx] = pred if ok else None
        else:
            llm_pred[idx] = float(sup_scores[idx])

    finals = rank_fusion(cand, arch_vecs, emb, llm_pred, args.no_llm, sup_rank=sup_rank, e_ref=e_ref)

    cand_set = set(cand)
    def unified_score(idx):
        return finals[idx] if idx in cand_set else (size + 1 - sup_rank[idx]) / size
    order = sorted(range(size), key=lambda i: (unified_score(i), llm_pred.get(i) or 0.0), reverse=True)
    llm_valid = [i for i in cand if llm_pred.get(i) is not None]
    llm_valid_sorted = sorted(llm_valid, key=lambda x: llm_pred[x], reverse=True)
    frank = {idx: pos + 1 for pos, idx in enumerate(llm_valid_sorted)}

    query_order = sorted(cand, key=lambda i: finals[i], reverse=True)[:args.query_topk]
    queried = []
    for idx in query_order:
        structure = alphas_to_structure(pop[idx].arch_parameters[0], pop[idx].arch_parameters[1])
        gstr = structure.tostr()
        if hall_of_fame is not None and pop[idx] is hall_of_fame["ind"]:
            valid_acc, test_acc = hall_of_fame["true_acc"], hall_of_fame["test_acc"]
            reused = True
        else:
            valid_acc, test_acc = query_nasbench_true_acc101(api, structure, mode = args.query_mode)
            QUERY_COUNT += 1
            reused = False
        pred = llm_pred.get(idx)
        err = abs(pred - valid_acc) if pred is not None else None
        q = {"idx": idx, "genotype_str": gstr, "arch_vector": arch_vecs[idx].tolist(),
             "pred_acc": pred, "true_acc": valid_acc, "test_acc": test_acc, "error": err}
        queried.append(q)
        if best_by_test is None or test_acc > best_by_test["test_acc"]:
            best_by_test = {"gen": gen, "genotype_str": gstr, "valid_acc": valid_acc, "test_acc": test_acc}
        logging.info("[INFO] Query #{}/{}: {} true={:.6f} llm={} err={}{}".format(
            QUERY_COUNT, args.query_topk * args.epochs, gstr, valid_acc,
            "-" if pred is None else "{:.6f}".format(pred),
            "-" if err is None else "{:.6f}".format(err),
            " (reused hof truth, no query)" if reused else ""))

    for q in queried:
        if not any(p["genotype_str"] == q["genotype_str"] for p in truth_pool):
            truth_pool.append({"genotype_str": q["genotype_str"], "true_acc_valid": q["true_acc"]})

    new_entries = [{k: q[k] for k in ("arch_vector", "genotype_str", "pred_acc", "true_acc", "error")}
                   for q in queried if q["error"] is not None]
    if new_entries:
        emb = update_emb(emb, new_entries, cap=args.emb_cap, threshold=args.sim_threshold)
        logging.info("[INFO] EMB updated -> size {} (+{} this gen)".format(len(emb), len(new_entries)))

    champion = max(queried, key=lambda q: q["true_acc"]) if queried else None

    for q in queried:
        if hall_of_fame is None or q["true_acc"] > hall_of_fame["true_acc"]:
            hall_of_fame = {"ind": pop[q["idx"]], "true_acc": q["true_acc"],
                            "test_acc": q["test_acc"], "genotype_str": q["genotype_str"]}

    if queried:
        for q in queried:
            pop[q["idx"]].set_fitness(0.0, q["true_acc"] * 100.0, q["test_acc"] * 100.0)
        queried_idx = [q["idx"] for q in sorted(queried, key=lambda x: x["true_acc"], reverse=True)]
        queried_set = set(queried_idx)
        final_order = queried_idx + [i for i in order if i not in queried_set]
    else:
        final_order = order
    mrank = {idx: pos + 1 for pos, idx in enumerate(final_order)}

    new_pop_list = [pop[idx] for idx in final_order]
    pop_list = population.get_population()
    pop_list.clear()
    pop_list.extend(new_pop_list)

    if hall_of_fame is not None:
        hof_ind = hall_of_fame["ind"]
        cur_pop = population.get_population()
        if hof_ind in cur_pop:
            cur_pop.remove(hof_ind)
            cur_pop.insert(0, hof_ind)
        else:
            cur_pop.pop()
            cur_pop.insert(0, hof_ind)
        logging.info("[INFO] Hall of fame anchored at #1: {} (true {:.6f} / test {:.6f})".format(
            hall_of_fame["genotype_str"], hall_of_fame["true_acc"], hall_of_fame["test_acc"]))

    if not args.no_debug_print:
        logging.info("[DEBUG] === New population after selection & fusion-sort: {} archs ===".format(size))
        best_row_50 = None
        for pos, (orig_idx, ind) in enumerate(zip(final_order, new_pop_list), start=1):
            struct_tmp = alphas_to_structure(ind.arch_parameters[0], ind.arch_parameters[1])
            gstr = struct_tmp.tostr()
            v_acc, t_acc = query_nasbench_true_acc101(api, struct_tmp, mode = args.query_mode)
            if orig_idx in cand_set:
                fus_s = "{:.4f}".format(finals[orig_idx])
                fr_s = "{}/{}".format(frank[orig_idx], len(llm_valid)) if orig_idx in frank else "----"
            else:
                fus_s, fr_s = "----", "----"
            pred_tmp = llm_pred.get(orig_idx)
            llm_s = "{:.6f}".format(pred_tmp) if pred_tmp is not None else "----"
            emb_cos = emb_max_sim(emb, arch_vecs[orig_idx])
            emb_cos_s = "{:.4f}".format(emb_cos) if emb else "----"
            logging.info("[DEBUG] {:3d} | supernet={:8.4f} | llm={:>8} | fusion={:>6} | frank={:>4} | mrank={:3d} | emb_cos={:>6} | truth_valid={:.6f} | truth_test={:.6f} | {}".format(
                pos, sup_scores[orig_idx], llm_s, fus_s, fr_s, mrank[orig_idx], emb_cos_s, v_acc, t_acc, gstr))
            if best_row_50 is None or v_acc > best_row_50[2]:
                best_row_50 = (pos, gstr, v_acc, t_acc)
        if best_row_50 is not None:
            logging.info("[DEBUG] BEST among these {} (by truth): pos={} | true_valid={:.6f} | test={:.6f} | {}".format(
                size, best_row_50[0], best_row_50[2], best_row_50[3], best_row_50[1]))
    return champion

DIR = "search-{}-nas101".format(time.strftime("%Y%m%d-%H%M%S"))
if args.dir is not None:
    if not os.path.exists(args.dir):
        utils.create_exp_dir(args.dir)
    DIR = os.path.join(args.dir, DIR)
else:
    DIR = os.path.join(os.getcwd(), DIR)
utils.create_exp_dir(DIR)
utils.create_exp_dir(os.path.join(DIR, "weights"))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

writer = SummaryWriter(os.path.join(DIR, 'runs'))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda:{}".format(args.gpu))
cpu_device = torch.device("cpu")
torch.cuda.set_device(args.gpu)
cudnn.deterministic = True
cudnn.enabled = True
cudnn.benchmark = False

assert args.api_path is not None, 'NAS-Bench-101 data path has not been provided'
from nasbench import api as nasbench_api
api = nasbench_api.NASBench(args.api_path)
logging.info("NAS-Bench-101 loaded from {}".format(args.api_path))

from config_utils import load_config
from datasets import get_datasets, get_nas_search_loaders

if args.cutout:
    train_data, valid_data, xshape, num_classes = get_datasets(name='cifar10', root=args.data, cutout=args.cutout)
else:
    train_data, valid_data, xshape, num_classes = get_datasets(name='cifar10', root=args.data, cutout=-1)
logging.info("train data len: {}, valid data len: {}, xshape: {}, #classes: {}".format(
    len(train_data), len(valid_data), xshape, num_classes))

config = load_config(path=args.config_path, extra={'class_num': num_classes, 'xshape': xshape}, logger=None)
_, train_loader, valid_loader = get_nas_search_loaders(
    train_data=train_data, valid_data=valid_data, dataset='cifar10',
    config_root='configs', batch_size=(args.batch_size, args.valid_batch_size), workers=args.workers)
train_queue, valid_queue = train_loader, valid_loader

model = TinyNetwork(C=args.init_channels, N=args.num_cells, num_classes=num_classes,
                    affine=False, track_running_stats=args.track_running_stats,
                    edge_relax=args.edge_relax)
model = model.to(device)

optimizer, _, criterion = get_optim_scheduler(parameters=model.get_weights(), config=config)
criterion = criterion.cuda()
logging.info(f'optimizer: {optimizer}\nCriterion: {criterion}')

best_arch_per_epoch = []
logging.info("[INFO] Initial architecture: {}".format(model.genotype().tostr()))

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

population = Population(pop_size=args.pop_size, num_nodes=NUM_INTERMEDIATE,
                        num_ops=NUM_OPS, num_edges=NUM_EDGES, device=device)
logging.info("args = %s", args)

de = DifferentialEvolution(args.pop_size, args.tsize, device,
                           NUM_INTERMEDIATE, NUM_OPS, NUM_EDGES, args.mutate_rate)

with open(args.fewshot_path, encoding='utf-8') as f:
    fewshot_examples = json.load(f)
logging.info("[INFO] few-shot examples: {}".format(len(fewshot_examples)))

llm_api_key = args.llm_api_key or os.environ.get('DEEPSEEK_API_KEY', '') or DEEPSEEK_API_KEY
if not args.no_llm and not llm_api_key:
    logging.error("[ERROR] No DeepSeek API key. Set --llm_api_key or env DEEPSEEK_API_KEY (or --no_llm).")
    sys.exit(1)
client = create_client(llm_api_key) if not args.no_llm else None
logging.info("[INFO] LLM mode: {}".format('enabled' if not args.no_llm else 'DISABLED (--no_llm)'))

emb = []
hall_of_fame = None
truth_pool = []
best_by_test = None      
QUERY_COUNT = 0
LLM_CALL_COUNT = 0

e_ref = 0.05
if not args.no_llm:
    calib_errs = []
    for ex in fewshot_examples:
        pred, ok = call_llm(client, build_emb_prompt101(fewshot_examples, [], ex["genotype_str"]))
        LLM_CALL_COUNT += 1
        if ok:
            calib_errs.append(abs(pred - ex["true_acc_valid"]))
    if calib_errs:
        e_ref = float(np.median(calib_errs))
    logging.info("[INFO] e_ref calibrated on initial example set: {:.6f} ({} samples)".format(e_ref, len(calib_errs)))

start = time.time()
for epoch in range(args.epochs):
    logging.info("[INFO] Generation {} training with learning rate {}".format(epoch + 1, scheduler.get_lr()[0]))
    start_time = time.time()

    train(model, train_queue, criterion, optimizer, epoch + 1)
    logging.info("[INFO] Training finished in {} minutes".format((time.time() - start_time) / 60))
    torch.save(model.state_dict(), "model.pt")
    scheduler.step()

    logging.info("[INFO] Evaluating Generation {} ".format(epoch + 1))
    champion = validation(model, valid_queue, criterion, epoch + 1)

    for i, p in enumerate(population.get_population()):
        writer.add_scalar("pop_top1_{}".format(i + 1), p.get_fitness(), epoch + 1)
        writer.add_scalar("pop_top5_{}".format(i + 1), p.top5.avg, epoch + 1)

    tmp = [(individual.arch_parameters[0].cpu().numpy(), individual.arch_parameters[1].cpu().numpy(),
            individual.get_fitness()) for individual in population.get_population()]
    with open(os.path.join(DIR, "population_{}.pickle".format(epoch + 1)), 'wb') as f:
        pickle.dump(tmp, f)

    if champion is not None:
        best_arch_per_epoch.append({"gen": epoch + 1, "genotype_str": champion["genotype_str"],
                                    "true_acc_valid": champion["true_acc"],
                                    "true_acc_test": champion["test_acc"],
                                    "llm_pred": champion["pred_acc"]})
        writer.add_scalar("test_acc", champion["test_acc"], epoch + 1)
        writer.add_scalar("valid_acc", champion["true_acc"], epoch + 1)
    else:
        best_arch_per_epoch.append({"gen": epoch + 1, "genotype_str": None,
                                    "true_acc_valid": None, "true_acc_test": None, "llm_pred": None})

    pop = de.evolve(population)
    population = pop

    last = time.time() - start_time
    logging.info("[INFO] {}/{} epoch finished in {} minutes".format(epoch + 1, args.epochs, last / 60))
    utils.save(model, os.path.join(DIR, "weights", "weights.pt"))

writer.close()

if emb:
    emb_path = os.path.join(DIR, "emb_final.json")
    with open(emb_path, 'w', encoding = 'utf-8') as f:
        json.dump(emb, f, indent = 2, ensure_ascii = False, default = float)
    logging.info("[INFO] Final EMB saved ({} entries) to {}".format(len(emb), emb_path))

last = time.time() - start
logging.info("[INFO] {} hours".format(last / 3600))

global_best = None
for entry in best_arch_per_epoch:
    if entry.get("true_acc_valid") is not None:
        if global_best is None or entry["true_acc_valid"] > global_best["true_acc_valid"]:
            global_best = entry
if global_best is not None:
    logging.info("[INFO] BEST (by valid, the search objective) at gen {}: valid {:.6f} / test {:.6f}".format(
        global_best["gen"], global_best["true_acc_valid"], global_best["true_acc_test"]))
    logging.info("[INFO]   genotype: {}".format(global_best["genotype_str"]))
if hall_of_fame is not None:
    logging.info("[INFO] Hall of fame (best valid ever): valid {:.6f} / test {:.6f}".format(
        hall_of_fame["true_acc"], hall_of_fame["test_acc"]))
    logging.info("[INFO]   genotype: {}".format(hall_of_fame["genotype_str"]))
if best_by_test is not None:
    logging.info("[INFO] BEST architecture (by test): gen {} | valid {:.6f} / test {:.6f}".format(
        best_by_test["gen"], best_by_test["valid_acc"], best_by_test["test_acc"]))
    logging.info("[INFO]   genotype: {}".format(best_by_test["genotype_str"]))
