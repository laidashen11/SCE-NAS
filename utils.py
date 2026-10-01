import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import pickle
import population
import genotypes
import torch.nn.functional as F

class AvgrageMeter(object):

	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0/batch_size))
	return res
'''
def accuracy2(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)
	print("[INFO] topk: {}, maxk: {}, batch_size: {}".format(topk, maxk, batch_size))

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0/batch_size))
	return res
'''
class Cutout(object):
	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)

		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img

def _data_transforms_cifar10(args):
	CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
	CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	if args.cutout:
		train_transform.transforms.append(Cutout(args.cutout_length))

	valid_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
		])
	return train_transform, valid_transform

def _data_transforms_cifar100(args):
	CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
	CIFAR_STD = [0.2675, 0.2565, 0.2761]

	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
	])
	if args.cutout:
		train_transform.transforms.append(Cutout(args.cutout_length))

	valid_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
		])
	return train_transform, valid_transform

def count_parameters_in_MB(model):
	return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save_checkpoint(state, is_best, save):
	filename = os.path.join(save, 'checkpoint.pth.tar')
	torch.save(state, filename)
	if is_best:
		best_filename = os.path.join(save, 'model_best.pth.tar')
		shutil.copyfile(filename, best_filename)

def save(model, model_path):
	torch.save(model.state_dict(), model_path)

def load(model, model_path, gpu = 0):
	model.load_state_dict(torch.load(model_path, map_location = 'cuda:{}'.format(gpu)), strict=False)

def drop_path(x, drop_prob):
	if drop_prob > 0.:
		keep_prob = 1.-drop_prob
		mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
		x.div_(keep_prob)
		x.mul_(mask)
	return x

def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.mkdir(path)
	print('Experiment dir : {}'.format(path))

	if scripts_to_save is not None:
		os.mkdir(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)

def check_equality(model, parameters):
	
	for x, y in zip(model.arch_parameters(), parameters):
		if not torch.all(x.eq(y)):
			return False
	return True

'''
def model_copy(model, criterion, device):
	model_copy = Network(model._C, model._num_classes, model._layers, criterion, device)
	#model_cpu = Network(model._C, model._num_classes, model._layers, criterion, device = torch.device("cpu"))
	
	#model_cpu.load_state_dict(model.state_dict())
	model_copy.load_state_dict(model.state_dict())
	model_copy.to(device)
	for x, y in zip(model_copy.arch_parameters(), model.arch_parameters()):
		x.data.copy_(y.data)

	return model_copy
'''

def load_population(file_name):
	pop = population.Population(0, 0)
	with open(file_name, 'rb') as f:
		pop = pickle.load(f)
	return pop

def derive_architecture(alphas):
	genotype = genotypes.PRIMITIVES
	ops = []
	for alpha in alphas:
		tmp = F.softmax(alpha, dim = -1)
		index = torch.argmax(tmp)
		op = genotype[index]
		if op == 'none':
			index = torch.topk(tmp, 2).indices[1]
			op = genotype[index]
		ops.append((op, tmp[index]))

	return ops

def derive_architecture_topk(alphas, k = 2):
	genotype = genotypes.PRIMITIVES
	num = alphas.size()[0]
	i = 1
	while i <= num:
		tmp = ((i + 1) * (i + 2)) / 2
		if ((tmp - 1) == num):
			steps = i
			break
		i += 1

	ops = []
	offset = 0
	for i in range(steps):
		t = []
		for j in range(2 + i):
			tmp = F.softmax(alphas[offset + j], dim = -1)
			index = torch.argmax(tmp)
			op = genotype[index]
			if op == 'none':
				index = torch.topk(tmp, 2).indices[1]
				op = genotype[index]
			t.append((op, j, tmp[index]))
		
		offset += j + 1
		t.sort(key = lambda x : x[2], reverse = True)
		for ip in range(k):
			p = t[ip]
			ops.append(tuple([p[0], p[1]]))
	return ops

def discretize(alphas, device):
	genotype = genotypes.PRIMITIVES
	normal_cell = derive_architecture_topk(alphas[0])
	reduction_cell = derive_architecture_topk(alphas[1])

	num = alphas[0].size()[0]
	i = 1
	while i <= num:
		tmp = ((i + 1) * (i + 2)) / 2
		if ((tmp - 1) == num):
			steps = i
			break
		i += 1

	new_alphas = []
	new_normal = torch.zeros_like(alphas[0]).to(device)
	new_reduction = torch.zeros_like(alphas[1]).to(device)
	
	i = 0
	offset = 0
	while i < len(normal_cell):
		op, cell = normal_cell[i]
		index = genotype.index(op)
		new_normal[int(offset + cell)][index] = 1
		
		i += 1
		op, cell = normal_cell[i]
		index = genotype.index(op)
		new_normal[int(offset + cell)][index] = 1
		offset += (i // 2) + 2
		i += 1

	i = 0
	offset = 0
	while i < len(reduction_cell):
		op, cell = reduction_cell[i]
		index = genotype.index(op)
		new_reduction[int(offset + cell)][index] = 1
		
		i += 1
		op, cell = reduction_cell[i]
		index = genotype.index(op)
		new_reduction[int(offset + cell)][index] = 1
		offset += (i // 2) + 2
		i += 1
	
	new_alphas = [new_normal, new_reduction]
	return new_alphas

def discretize2(alphas, beta, device):
	genotype = genotypes.PRIMITIVES
	normal_cell = derive_architecture_topk(alphas[0])
	reduction_cell = derive_architecture_topk(alphas[1])

	num = alphas[0].size()[0]
	i = 1
	while i <= num:
		tmp = ((i + 1) * (i + 2)) / 2
		if ((tmp - 1) == num):
			steps = i
			break
		i += 1

	new_alphas = []
	new_normal = torch.zeros_like(alphas[0]).to(device)
	new_reduction = torch.zeros_like(alphas[1]).to(device)
	
	i = 0
	offset = 0
	while i < len(normal_cell):
		op, cell = normal_cell[i]
		index = genotype.index(op)
		new_normal[int(offset + cell)][index] = beta
		
		i += 1
		op, cell = normal_cell[i]
		index = genotype.index(op)
		new_normal[int(offset + cell)][index] = beta
		offset += (i // 2) + 2
		i += 1

	i = 0
	offset = 0
	while i < len(reduction_cell):
		op, cell = reduction_cell[i]
		index = genotype.index(op)
		new_reduction[int(offset + cell)][index] = beta
		
		i += 1
		op, cell = reduction_cell[i]
		index = genotype.index(op)
		new_reduction[int(offset + cell)][index] = beta
		offset += (i // 2) + 2
		i += 1
	
	new_alphas = [new_normal, new_reduction]
	return new_alphas

