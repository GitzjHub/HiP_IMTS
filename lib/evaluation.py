import torch
from tqdm import tqdm
import lib.utils as utils

def compute_error(truth, pred_y, mask, func, reduce):

	if len(pred_y.shape) == 3: 
		pred_y = pred_y.unsqueeze(dim=0)
	n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()
	truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
	mask = mask.repeat(pred_y.size(0), 1, 1, 1)

	if(func == "MSE"):
		error = ((truth_repeated - pred_y)**2) * mask
	elif(func == "MAE"):
		error = torch.abs(truth_repeated - pred_y) * mask
	else:
		raise Exception("Error function not specified")

	error_var_sum = error.reshape(-1, n_dim).sum(dim=0)
	mask_count = mask.reshape(-1, n_dim).sum(dim=0)

	if reduce == "mean":
		error_var_avg = error_var_sum / (mask_count + 1e-8)
		n_avai_var = torch.count_nonzero(mask_count)
		error_avg = error_var_avg.sum() / n_avai_var
		return error_avg
	elif reduce == "sum":
		return error_var_sum, mask_count

def compute_all_losses(model, batch_dict):

	pred_y = model.forecasting(batch_dict["tp_to_predict"], 
		batch_dict["observed_data_List"], batch_dict["observed_tp_List"],
		batch_dict["observed_mask_List"])

	# Compute avg error of each variable first, then compute avg error of all variables
	mse = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MSE", reduce="mean")
	mae = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MAE", reduce="mean")

	loss = mse

	results = {}
	results["loss"] = loss
	results["mse"] = mse.item()
	results["mae"] = mae.item()

	return results

def evaluation(model, dataloader, n_batches):

	n_eval_samples = 0

	total_results = {
		"mse": 0, "mae": 0,
	}
	norm_dict = {"data_max": 0, "data_min": 0}
	for i in tqdm(range(n_batches)):

		batch_dict = utils.get_next_batch(dataloader)
		norm_dict["data_max"] = batch_dict["data_max"]
		norm_dict["data_min"] = batch_dict["data_min"]

		pred_y = model.forecasting(batch_dict["tp_to_predict"], 
			batch_dict["observed_data_List"], batch_dict["observed_tp_List"],
			batch_dict["observed_mask_List"])

		se_var_sum, mask_count = compute_error(batch_dict["data_to_predict"], pred_y, mask=batch_dict["mask_predicted_data"], func="MSE", reduce="sum")

		ae_var_sum, _ = compute_error(batch_dict["data_to_predict"], pred_y, mask = batch_dict["mask_predicted_data"], func="MAE", reduce="sum")

		total_results["mse"] += se_var_sum
		total_results["mae"] += ae_var_sum

		n_eval_samples += mask_count

	n_avai_var = torch.count_nonzero(n_eval_samples)

	total_results["mse"] = (total_results["mse"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mae"] = (total_results["mae"] / (n_eval_samples + 1e-8)).sum() / n_avai_var

	for key, var in total_results.items(): 
		if isinstance(var, torch.Tensor):
			var = var.item()
		total_results[key] = var

	return total_results

