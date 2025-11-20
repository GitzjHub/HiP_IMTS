import math
import torch.nn.functional as F
from Transformer_EncDec import EncoderLayer,Encoder
from SelfAttention_Family import DifferenceFormerlayer
from Difference_Pre import DifferenceDataEmb, DataRestoration
from Frequency_Embed import Frequency_Embedding
from lib.evaluation import *
import torch.nn as nn

class nconv(nn.Module):
	def __init__(self):
		super(nconv, self).__init__()

	def forward(self, x, A):
		x = torch.einsum('bfnm,bmnv->bfvm', (x, A))  # used
		return x.contiguous()

class linear(nn.Module):
	def __init__(self, c_in, c_out):
		super(linear, self).__init__()
		self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

	def forward(self, x):
		return self.mlp(x)

class gcn(nn.Module):
	def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
		super(gcn, self).__init__()
		self.nconv = nconv()
		c_in = (order * support_len + 1) * c_in
		self.mlp = linear(c_in, c_out)
		self.dropout = dropout
		self.order = order

	def forward(self, x, support):
		out = [x]
		for a in support:
			x1 = self.nconv(x, a)
			out.append(x1)
			for k in range(2, self.order + 1):
				x2 = self.nconv(x1, a)
				out.append(x2)
				x1 = x2

		h = torch.cat(out, dim=1)
		h = self.mlp(h)
		return F.relu(h)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class s_PatchModel(nn.Module):
	def __init__(self, args, index, supports = None):
	
		super(s_PatchModel, self).__init__()
		self.device = args.device
		self.hid_dim = args.hid_dim
		self.N = args.ndim
		self.M = args.npatches[index]
		self.batch_size = None
		self.n_layer = args.nlayer
		self.dropout = args.dropout
		self.activation = args.activation
		self.n_heads = args.nhead
		self.supports = supports

		## DTCN
		input_dim = 1 + args.te_dim
		dtcn_dim = self.hid_dim - 1
		self.dtcn_dim = dtcn_dim
		self.Filter_Generators = nn.Sequential(
				nn.Linear(input_dim, dtcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(dtcn_dim, dtcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(dtcn_dim, input_dim*dtcn_dim, bias=True))
		self.T_bias = nn.Parameter(torch.randn(1, dtcn_dim))

		self.d_ff = self.hid_dim
		## Difference Attention Network
		self.diff_data_emb = nn.ModuleList()
		self.difference_attention = nn.ModuleList()
		self.data_restoration = nn.ModuleList()
		for _ in range(self.n_layer):
			self.diff_data_emb.append(DifferenceDataEmb(self.hid_dim))
			self.difference_attention.append(Encoder(
				[
					EncoderLayer(
						DifferenceFormerlayer(
							self.hid_dim,
							self.n_heads,
							self.dropout
						),
						self.hid_dim,
						self.d_ff,
						dropout=self.dropout,
						activation=self.activation,
					)
					for _ in range(args.elayer)
				],
				norm_layer=torch.nn.LayerNorm(self.hid_dim)))
			self.data_restoration.append(DataRestoration(self.hid_dim))

		## Frequency Convolution Network
		self.freq_embedding = nn.ModuleList()
		for _ in range(self.n_layer):
			self.freq_embedding.append(Frequency_Embedding(self.M, self.hid_dim))

		### Inter-time series modeling ###
		self.supports_len = 0
		if supports is not None:
			self.supports_len += len(supports)
		if supports is None:
			self.supports = []
		self.nodevec_dim = args.node_dim

		self.nodevec1 = nn.Parameter(torch.randn(self.N, self.nodevec_dim ).cuda(), requires_grad=True)
		self.nodevec2 = nn.Parameter(torch.randn(self.nodevec_dim , self.N).cuda(), requires_grad=True)

		self.nodevec_linear1 = nn.ModuleList()
		self.nodevec_linear2 = nn.ModuleList()
		self.nodevec_gate1 = nn.ModuleList()
		self.nodevec_gate2 = nn.ModuleList()
		for _ in range(self.n_layer):
			self.nodevec_linear1.append(nn.Linear(self.hid_dim, self.nodevec_dim))
			self.nodevec_linear2.append(nn.Linear(self.hid_dim, self.nodevec_dim ))
			self.nodevec_gate1.append(nn.Sequential(
				nn.Linear(self.hid_dim+self.nodevec_dim , 1),
				nn.Tanh(),
				nn.ReLU()))
			self.nodevec_gate2.append(nn.Sequential(
				nn.Linear(self.hid_dim+self.nodevec_dim , 1),
				nn.Tanh(),
				nn.ReLU()))

		self.supports_len += 1

		self.gconv = nn.ModuleList() # gragh conv
		for _ in range(self.n_layer):
			self.gconv.append(gcn(self.hid_dim, self.hid_dim, self.dropout, support_len=self.supports_len, order=args.hop))

		### Encoder output layer ###
		self.outlayer = args.outlayer
		if (self.outlayer == "Linear"):
			self.temporal_agg = nn.Sequential(
				nn.Linear(self.hid_dim * self.M, self.hid_dim))

		elif (self.outlayer == "CNN"):
			self.temporal_agg = nn.Sequential(
				nn.Conv1d(self.hid_dim, self.hid_dim, kernel_size=self.M))
	
	def DTCN(self, X_int, mask_X):
		N, Lx, _ = mask_X.shape
		Filter = self.Filter_Generators(X_int)
		Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
		# normalize along with sequence dimension
		Filter_seqnorm = F.softmax(Filter_mask, dim=-2)
		Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.dtcn_dim, -1)
		X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.dtcn_dim, 1)
		ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1)
		h_t = torch.relu(ttcn_out + self.T_bias)
		return h_t

	def IMTS_Model(self, x, mask_X, index):
		# mask for the patch
		mask_patch = (mask_X.sum(dim=1) > 0)

		### DTCN for patch modeling ###
		x_patch = self.DTCN(x, mask_X)
		x_patch = torch.cat([x_patch, mask_patch],dim=-1)
		x_patch = x_patch.view(self.batch_size, self.N, self.M, -1)
		B, N, M, D = x_patch.shape

		x = x_patch
		x = x.reshape(B*N, M, -1)

		for layer in range(self.n_layer):

			if (layer > 0):  # residual
				x_last = x.clone()
			x = x.reshape(B*N, M, -1)

			# difference attention network
			x_diff_emb, init_vals = self.diff_data_emb[layer](x)
			x_diff_enc, _ = self.difference_attention[layer](x_diff_emb)
			enc_out_1 = self.data_restoration[layer](x_diff_enc, init_vals)

			# frequency convolution network
			enc_out_2 = self.freq_embedding[layer](x)

			x = enc_out_1 + enc_out_2
			x = x.view(x_patch.shape)

			### GNN for inter-time series modeling ###
			### time-adaptive graph structure learning ###
			nodevec1 = self.nodevec1.view(1, 1, N, self.nodevec_dim).repeat(B, M, 1, 1)
			nodevec2 = self.nodevec2.view(1, 1, self.nodevec_dim, N).repeat(B, M, 1, 1)
			x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
			x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))
			x_p1 = x_gate1 * self.nodevec_linear1[layer](x)
			x_p2 = x_gate2 * self.nodevec_linear2[layer](x)
			nodevec1 = nodevec1 + x_p1.permute(0, 2, 1, 3)
			nodevec2 = nodevec2 + x_p2.permute(0, 2, 3, 1)

			adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1)
			new_supports = self.supports + [adp]

			x = self.gconv[layer](x.permute(0, 3, 1, 2), new_supports)
			x = x.permute(0, 2, 3, 1)

			if (layer > 0):  # residual addition
				x = x_last + x

		x_out = x
		### Output layer ###
		if (self.outlayer == "CNN"):
			x_out = x_out.reshape(self.batch_size * self.N, self.M, -1).permute(0, 2, 1)
			x_out = self.temporal_agg(x_out)
			x_out = x_out.view(self.batch_size, self.N, -1)

		elif (self.outlayer == "Linear"):
			x_out = x_out.reshape(self.batch_size, self.N, -1)
			x_out = self.temporal_agg(x_out)

		return x_out


class HiP_IMTS(nn.Module):
	def __init__(self, args):
		super(HiP_IMTS, self).__init__()
		self.nscale = args.nscale
		self.batch_size = None

		self.m_PatchModel = nn.ModuleList()
		for index in range(self.nscale):
			self.m_PatchModel.append(s_PatchModel(args, index))
		self.m_PatchModel = self.m_PatchModel.to(args.device)

		## Time embedding
		self.te_scale = nn.Linear(1, 1)
		self.te_periodic = nn.Linear(1, args.te_dim - 1)

		### Decoder ###
		self.decoder = nn.ModuleList()
		for index in range(self.nscale):
			self.decoder.append(nn.Sequential(
				nn.Linear(args.hid_dim + args.te_dim, args.hid_dim),
				nn.ReLU(inplace=True),
				nn.Linear(args.hid_dim, args.hid_dim),
				nn.ReLU(inplace=True),
				nn.Linear(args.hid_dim, 1)
			))
		self.decoder = self.decoder.to(args.device)

	def LearnableTE(self, tt):
		out1 = self.te_scale(tt)
		out2 = torch.sin(self.te_periodic(tt))
		return torch.cat([out1, out2], -1)

	def forecasting(self, time_steps_to_predict, X_List, truth_time_steps_List, mask_List=None):
		o_list = []
		for index in range(self.nscale):
			X = X_List[index].clone()
			truth_time_steps = truth_time_steps_List[index].clone()
			mask = mask_List[index].clone()

			B, M, L_in, N = X.shape
			self.batch_size = B
			self.m_PatchModel[index].batch_size = B
			X = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
			truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
			mask = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
			te_his = self.LearnableTE(truth_time_steps)

			X = torch.cat([X, te_his], dim=-1)

			### *** a encoder to model irregular time series
			hs = self.m_PatchModel[index].IMTS_Model(X, mask, index)

			### Decoder
			L_pred = time_steps_to_predict.shape[-1]
			hs = hs.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)
			tp_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
			te_pred = self.LearnableTE(tp_to_predict)

			hs = torch.cat([hs, te_pred], dim=-1)

			output = self.decoder[index](hs).permute(3, 0, 2, 1)
			o_list.append(output)

		output = torch.mean(torch.stack(o_list), dim=0)
		return output
