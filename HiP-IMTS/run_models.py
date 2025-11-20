import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
import torch
import torch.nn as nn
import torch.optim as optim
from lib.parse_datasets import parse_datasets
from model.HiP_IMTS import *

parser = argparse.ArgumentParser('IMTS Forecasting')

parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--elayer', type=int, default=1, help="# of layer in Difference Attention Network")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in GNN")
parser.add_argument('--epoch', type=int, default=100, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=12, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('-ps', '--patch_sizes', type=float, default=[2,4], help="window size for a patch")
parser.add_argument('--logmode', type=str, default="w", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=0.001, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=1e-3, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: activity, physionet, mimic, ushcn")

# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='tPatchGNN', help="Model name")
parser.add_argument('--outlayer', type=str, default='Linear', help="Model name")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument("--activation", type=str, default="gelu", help="activation")

parser.add_argument("--train", type=str, default=True, help="Is training?")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')


if __name__ == '__main__':

	args = parser.parse_args()
	args.seed = random.randint(1, 30)
	utils.setup_seed(args.seed)
	args.strides = args.patch_sizes
	args.nscale = len(args.patch_sizes)
	args.npatches = []
	for i in range(args.nscale):
		args.npatches.append(
			int(np.ceil((args.history - args.patch_sizes[i]) / args.strides[i])) + 1)  # (window size for a patch)
	print("number of npatches:", args.npatches)
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	file_name = os.path.basename(__file__)[:-3]
	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.PID = os.getpid()
	print("PID, device:", args.PID, args.device)
	print(f"Running dataset: {args.dataset} with seed: {args.seed}")

	### load dataset ###
	data_obj = parse_datasets(args, patch_ts=True)
	input_dim = data_obj["input_dim"]

	### Model setting ###
	args.ndim = input_dim
	model = HiP_IMTS(args).to(args.device)

	if(args.n < 12000):
		args.state = "debug"
		dir_path = "logs/{}_{}".format(args.dataset, args.state)
		log_path = "{}_{}.log".format(args.dataset, args.state)
		log_path = os.path.join(dir_path, log_path)
	else:
		dir_path = "logs/{}".format(args.dataset)
		if args.train == True:
			log_path = "{}_patch{}.log".format(args.dataset, args.patch_sizes)
			log_path = os.path.join(dir_path, log_path)
		else:
			log_path = "{}_patch{}_test.log".format(args.dataset, args.patch_sizes)
			log_path = os.path.join(dir_path, log_path)

	if not os.path.exists(dir_path):
		utils.makedirs(dir_path)
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(args)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	if args.train == True:
		best_val_mse = np.inf
		best_test_mse = np.inf
		best_test_mae = np.inf
		test_res = None
		loss_list = []
		test_loss_list = []
		val_loss_list = []

		for itr in range(args.epoch):
			print("epoch: ", itr)

			### Training ###
			model.train()
			train_loss = 0
			for _ in tqdm(range(num_batches)):
				optimizer.zero_grad()
				batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
				train_res = compute_all_losses(model, batch_dict)
				train_res["loss"].backward()
				train_loss += train_res["loss"]
				optimizer.step()
			train_loss = train_loss / num_batches
			loss_list.append(train_loss.item())

			## Validation ###
			model.eval()
			with torch.no_grad():

				val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
				val_loss_list.append(val_res["mse"])

				if(val_res["mse"] < best_val_mse):
					best_val_mse = val_res["mse"]
					best_iter = itr

					# save model
					model_save_path = os.path.join(dir_path, 'model.pth')
					torch.save(model.state_dict(), model_save_path)

					# Testing ###
					test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
					test_loss_list.append(test_res["mse"])

				logger.info("Train - Loss (one batch): {:.5f}".format(train_loss.item()))
				logger.info("Val - MSE = {:.5f}, MAE = {:.5f}".format(val_res["mse"], val_res["mae"]))
				if(test_res != None):
					logger.info("Best Test - epoch = {}, MSE = {:.5f}, MAE = {:.5f}".format(best_iter, test_res["mse"], test_res["mae"]))

			if(itr - best_iter >= args.patience):
				print("Exp has been early stopped!")
				sys.exit(0)


	else:
		model_save_path = os.path.join(dir_path, 'model.pth')
		model.load_state_dict(torch.load(model_save_path))
		model.eval()
		with torch.no_grad():

			tqdm.write("Start Testing......")
			test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
			print("Test - MSE = {:.5f}, MAE = {:.5f}".format(test_res["mse"], test_res["mae"]))
			logger.info("Test - MSE = {:.5f}, MAE = {:.5f}".format(test_res["mse"], test_res["mae"]))