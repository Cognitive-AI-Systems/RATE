import sys
import math
import functools

import numpy as np
 
import torch
import torch.nn as nn
import torch.nn.functional as F

from RATE_GTrXL.utils import LogUniformSampler, sample_logits, ProjectedAdaptiveLogSoftmax
from RATE_GTrXL.blocks import RelPartialLearnableDecoderLayer, PositionalEmbedding

class MemTransformerLM(nn.Module):
    def __init__(self, STATE_DIM, 
                 ACTION_DIM, 
                 n_token, 
                 n_layer, 
                 n_head,
                 n_head_ca, 
                 d_model, 
                 d_head, 
                 d_inner,
                 dropout, 
                 dropatt, 
                 tie_weight=True, 
                 d_embed=None, 
                 div_val=1, 
                 tie_projs=[False], 
                 pre_lnorm=False, ### ! LayerNorm(W)
                 tgt_len=None, 
                 ext_len=None, 
                 mem_len=None, 
                 num_mem_tokens=None, 
                 read_mem_from_cache=False, 
                 mem_at_end=True,
                 cutoffs=[], 
                 adapt_inp=False,
                 same_length=False, 
                 attn_type=0, 
                 clamp_len=-1, 
                 sample_softmax=-1,
                 max_ep_len=1000,
                 num_classes_ass_ret=16,
                 mode='mujoco',
                 use_gate=False,
                 use_stable_version=False,
                 mrv_act='relu'):
        
        super(MemTransformerLM, self).__init__()

        qkw_norm = pre_lnorm

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.STATE_DIM = STATE_DIM
        self.ACTION_DIM = ACTION_DIM
        self.mode = mode
        self.loss_last_coef = 1
        self.num_classes_ass_ret = num_classes_ass_ret
        self.flag = 0
        self.is_first_segment = True
        self.log_prob = 0
        self.buf = []
        
        self.embed_timestep = nn.Embedding(max_ep_len, d_embed)
        self.embed_ln = nn.LayerNorm(d_embed)
        self.ret_emb = nn.Linear(1, d_embed)

        if mrv_act == 'relu':
            self.mrv_act = F.relu
        elif mrv_act == 'leaky_relu':
            self.mrv_act = F.leaky_relu
        elif mrv_act == 'elu':
            self.mrv_act == F.elu
        elif mrv_act == 'tanh':
            self.mrv_act = F.tanh
        elif mrv_act == 'no_act':
            self.mrv_act = None
        else:
            raise NotImplementedError('This MRV activation is not studied')
        
        
        if self.mode in ['mujoco','tmaze', 'aar']:
            self.state_encoder = nn.Sequential(
                                    nn.Linear(self.STATE_DIM, d_embed), #RMT MUJOCO
                                    nn.ReLU())
            
        if self.mode in ['atari']:
            self.state_encoder = nn.Sequential(
                                 nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(3136, d_embed), nn.Tanh())
            
        if self.mode in ['key_to_door']:
            self.state_encoder = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.MaxPool2d(kernel_size=2, stride=2),

                                               nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(64, 1, kernel_size=2, padding=1),
                                               nn.ReLU(inplace=True),

                                               nn.Flatten(start_dim=2),
                                               nn.Linear(16, d_embed),
                                               nn.ReLU())     

        

        if self.mode == 'mujoco':
            self.head = nn.Sequential(*([nn.Linear(d_embed, self.ACTION_DIM)] + ([nn.Tanh()])))
            self.state_encoder = nn.Linear(self.STATE_DIM, d_embed)
            self.action_embeddings = nn.Linear(self.ACTION_DIM, d_embed)
            self.ret_emb = nn.Linear(1, d_embed)

        if self.mode == 'atari':
            self.head = nn.Linear(d_embed, n_token, bias=False)
            self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), 
                                               nn.ReLU(),
                                               nn.Conv2d(32, 64, 4, stride=2, padding=0), 
                                               nn.ReLU(),
                                               nn.Conv2d(64, 64, 3, stride=1, padding=0), 
                                               nn.ReLU(),
                                               nn.Flatten(), 
                                               nn.Linear(3136, d_embed), 
                                               nn.Tanh())
            self.action_embeddings = nn.Sequential(nn.Embedding(n_token, d_embed), nn.Tanh())
            self.ret_emb = nn.Sequential(nn.Linear(1, d_embed), nn.Tanh())
        
        if self.mode == 'key_to_door':
            self.head = nn.Sequential(*([nn.Linear(d_embed, 4)] + ([nn.Tanh()])))    



        if self.mode == 'doom':
            self.head = nn.Linear(d_embed, self.ACTION_DIM, bias=False)
            self.state_encoder = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4, padding=0),
                                           	nn.ReLU(),
                                           	nn.Conv2d(32, 64, 4, stride=2, padding=0),
                                           	nn.ReLU(),
                                           	nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                           	nn.ReLU(),
                                           	nn.Flatten(), nn.Linear(2560, d_embed),
                                           	nn.Tanh())
            self.action_embeddings = nn.Sequential(nn.Embedding(self.ACTION_DIM, d_embed), nn.Tanh())
            self.ret_emb = nn.Sequential(nn.Linear(1, d_embed), nn.Tanh())

        if self.mode == 'minigrid_memory':
            self.head = nn.Linear(d_embed, self.ACTION_DIM, bias=False)
            self.state_encoder = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4, padding=0),
                                           	nn.ReLU(),
                                           	nn.Conv2d(32, 64, 4, stride=2, padding=0),
                                           	nn.ReLU(),
                                           	nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                           	nn.ReLU(), 
                                           	nn.Flatten(), nn.Linear(3136, d_embed))
            self.action_embeddings = nn.Sequential(nn.Embedding(self.ACTION_DIM, d_embed))
            self.ret_emb = nn.Sequential(nn.Linear(1, d_embed))

        if self.mode == 'memory_maze':
            self.head = nn.Linear(d_embed, self.ACTION_DIM, bias=False)
            # * v9 /  v8_RATE_pad_2: model parameters: 2324656
            self.state_encoder = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4, padding=2),
                                    nn.ReLU(), 
                                    nn.Conv2d(32, 64, 4, stride=2, padding=2),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.Flatten(), nn.Linear(7744, d_embed), # 211 -> 4096 222 -> 7744 222+mp2 -> 1600
                                    nn.Tanh())

            self.action_embeddings = nn.Sequential(nn.Embedding(self.ACTION_DIM, d_embed), nn.Tanh())
            self.ret_emb = nn.Sequential(nn.Linear(1, d_embed), nn.Tanh()) 
        
        if self.mode == 'tmaze':
            self.head = nn.Linear(d_embed, 4, bias=False)
            self.action_embeddings = nn.Sequential(nn.Embedding(4+1, d_embed), nn.Tanh())
            self.ret_emb = nn.Linear(1, d_embed)
            self.state_encoder = nn.Linear(self.STATE_DIM, d_embed)

        if self.mode == 'aar':
            self.head = nn.Linear(d_embed, 3, bias=False)
            self.action_embeddings = nn.Sequential(nn.Embedding(3+1, d_embed), nn.Tanh())
            self.ret_emb = nn.Linear(1, d_embed)
            self.state_encoder = nn.Linear(self.STATE_DIM, d_embed)
            


        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = None#tgt_len + ext_len + mem_len + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.init_mem_tokens()
        self.read_mem_from_cache = read_mem_from_cache 
        self.mem_at_end = mem_at_end
        self.n_head_ca = n_head_ca

        if self.n_head_ca != 0:
            self.mha_mem_to_mem = MultiHeadAttention(d_q=self.d_model,
                                                    d_k=self.d_model, 
                                                    d_v=self.d_model, 
                                                    d_model=self.d_model, 
                                                    num_heads=self.n_head_ca,
                                                    dropout_p=dropatt,
                                                    is_causal=False)

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        use_gate=use_gate, use_stable_version=use_stable_version,
                        qkw_norm=qkw_norm,# !
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1: # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, 
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                        setattr(self.crit, f'out_projs_{i}', self.word_emb.emb_projs_0)
                    elif tie_proj and div_val != 1:
                        setattr(self.crit, f'out_projs_{i}', getattr(self.word_emb, f'emb_projs_{i}'))
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]
                        # self.crit.out_projs[i] = getattr(self.word_emb, f'emb_projs_{i}')

        self.same_length = same_length 
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention 
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
 
    def init_mems(self, device):
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=torch.float, device=device)
                mems.append(empty)

            return mems
        else:
            return None

    def init_mem_tokens(self):
        if self.num_mem_tokens == 0:
            self.mem_tokens = None
        else:
            mem_tokens = [torch.randn(1, self.d_model)] * self.num_mem_tokens
            mem_tokens = torch.cat(mem_tokens, dim=0).view(self.num_mem_tokens, 1, -1)
            mem_tokens = torch.nn.Parameter(mem_tokens, requires_grad=True)
            self.register_parameter(param=mem_tokens, name='mem_tokens')

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                # print(mems[i].shape, hids[i].shape, cat.shape, beg_idx, end_idx)
                new_mems.append(cat[beg_idx:end_idx].detach())
                
        return new_mems

    def _forward(self, word_emb, mems=None, mem_tokens=None):

        #word_emb = self.word_emb(dec_inp)
        bsz, qlen, _ = word_emb.size()
        # print(qlen)
        word_emb = word_emb.permute(1,0,2)

        mlen = mems[0].size(0) if mems is not None else 0
        # print(mlen)
        #mlen = 0
        # print("mlen1", mlen)
        #print(mem_tokens.shape, word_emb.shape)
        # Concat with mem_tokens
        if mem_tokens is not None:
            #print(mem_tokens.shape, word_emb.shape)
            #print(mem_tokens.shape, word_emb.shape, " Shapes here")
            word_emb = torch.cat((mem_tokens, word_emb), dim=0)
            # print(word_emb.shape)
            if self.mem_at_end:
                word_emb = torch.cat((word_emb, mem_tokens), dim=0) # shape num_mem_tokens + 3*context_length + num_mem_tokens, bs, emb_dim
                
        #print(word_emb.shape)

        qlen = word_emb.shape[0]
        klen = mlen + qlen
        
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()

            if self.num_mem_tokens != 0:
                dec_attn_mask[:self.num_mem_tokens, mlen:mlen+self.num_mem_tokens] = 0
                dec_attn_mask[:self.num_mem_tokens, :mlen] = 1 - int(self.read_mem_from_cache)
                if self.mem_at_end:
                    dec_attn_mask[-self.num_mem_tokens:, -self.num_mem_tokens:] = 0
                    dec_attn_mask[-self.num_mem_tokens:, :mlen] = 1 - int(self.read_mem_from_cache)
            dec_attn_mask = dec_attn_mask[:,:,None]
        hids = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)

            #print("klen", klen)
            #print("pos_seq", pos_seq.shape)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb) # self.drop(word_emb) word_emb
            pos_emb = self.drop(pos_emb) #self.drop(pos_emb) pos_emb


            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                # print('*****', mems_i.shape)
                core_out, self.attn_map = layer(core_out, pos_emb, self.r_w_bias,
                                              self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                
                hids.append(core_out)
                # print('!!!', core_out.shape)
        elif self.attn_type == 1: # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                        r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)
    
        new_mems = self._update_mems(hids, mems, qlen, mlen)             #(hids, mems, qlen, mlen) #(hids, mems, mlen, qlen) original
        # len(new_mems) = n_layer + 1, new_mems[i].shape = 9x64x32
        # print(core_out.shape, len(new_mems), new_mems[0].shape)
        return core_out, new_mems
    
    def forward(self, states, actions, rtgs, target, timesteps, *mems, mem_tokens=None, masks=None): # data
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        
        if self.mode in ['tmaze', 'aar']:
            ''' TMAZE MODE '''
            if not mems: mems = self.init_mems(states.device)
            state_embeddings = self.state_encoder(states) # (batch * block_size, n_embd)
            rtg_embeddings = self.ret_emb(rtgs)

            B = state_embeddings.shape[0]
            B1 = state_embeddings.shape[1]
            if actions is not None:
                use_long = False
                for name, module in self.action_embeddings.named_children():
                    if isinstance(module, nn.Embedding):
                        use_long = True
                if use_long:
                    if self.mode == 'tmaze':
                        actions = torch.where(actions == -10, torch.tensor(4), actions)
                    elif self.mode == 'aar':
                        actions = torch.where(actions == -10, torch.tensor(3), actions)
                    actions = actions.to(dtype=torch.long, device=states.device)
                    action_embeddings = self.action_embeddings(actions).squeeze(2) # (batch, block_size, n_embd)
                else:
                    action_embeddings = self.action_embeddings(actions) # (batch, block_size, n_embd)
                token_embeddings = torch.zeros((B, B1*3 - int(target is None), self.d_embed), dtype=torch.float32, device=state_embeddings.device)
                token_embeddings[:, ::3, :] = rtg_embeddings #+ time_embeddings
                token_embeddings[:, 1::3, :] = state_embeddings #+ time_embeddings
                token_embeddings[:, 2::3, :] = action_embeddings[:,-B1 + int(target is None):,:]

            else:
                token_embeddings = torch.zeros((B, B1*2, self.d_embed), dtype=torch.float32, device=state_embeddings.device)
                token_embeddings[:,::2,:] = rtg_embeddings
                token_embeddings[:,1::2,:] = state_embeddings

            hidden, new_mems = self._forward(token_embeddings, mems=mems, mem_tokens=mem_tokens) #hidden.shape = (total_len, bs, emb_dim) new_mems[i].shape = (MEM_LEN, bs, d_model)
            hidden = hidden.permute(1,0,2)
            num_mem = self.num_mem_tokens
            if self.num_mem_tokens > 0:
                if self.mem_at_end:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -num_mem:, :]
                else:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -tgt_len-num_mem:-tgt_len, :]

                if self.n_head_ca != 0:
                    # new_mem_tokens = F.relu(hidden[:, -num_mem:, :])
                    if self.mrv_act is not None:
                        new_mem_tokens = self.mrv_act(hidden[:, -num_mem:, :])
                    else:
                        new_mem_tokens = hidden[:, -num_mem:, :]

                    mem_tokens = mem_tokens.permute(1,0,2)
                    mask_mem_mem = torch.ones((new_mem_tokens.shape[1], new_mem_tokens.shape[1]), dtype=torch.bool).to(new_mem_tokens.device)
                    mem_tokens_write, _ = self.mha_mem_to_mem(mem_tokens, new_mem_tokens, new_mem_tokens, attn_mask=mask_mem_mem)

            if self.mem_at_end:
                logits = self.head(hidden)[:, num_mem:-num_mem]
            else:
                tgt_len = token_embeddings.shape[1] # total_len
                logits = self.head(hidden)[:, -tgt_len:] # was tgt_len # logits: torch.Size([64, 301, 4])
            
            if actions is not None:
                logits = logits[:, 1::3, :]
            else:
                logits = logits[:, 1:, :]
        
        else:
            if not mems: mems = self.init_mems(states.device)

            if self.mode == 'doom':
                B, B1, C, H, W = states.shape
                states = states.reshape(-1, C, H, W).type(torch.float32).contiguous() 
            elif self.mode == 'atari':
                if len(states.shape) == 5:
                    B, B1, C, H, W = states.shape
                elif len(states.shape) == 6:
                    B, B1, _, C, H, W = states.shape
                else:
                    B, B1, _ = states.shape
                states = states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous() 
            elif self.mode == 'memory_maze':
                B, B1, C, H, W = states.shape
                states = states.reshape(-1, C, H, W).type(torch.float32).contiguous() 
            elif self.mode == 'minigrid_memory':
                B, B1, C, H, W = states.shape
                states = states.reshape(-1, 3, 84, 84).type(torch.float32).contiguous() 
            else:
                B, B1, C = states.shape 

            state_embeddings = self.state_encoder(states) # (batch * block_size, n_embd)
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
            rtg_embeddings = self.ret_emb(rtgs)
            # time_embeddings = self.embed_timestep(timesteps)

            if actions is not None:
                use_long = False
                for name, module in self.action_embeddings.named_children():
                    if isinstance(module, nn.Embedding):
                        use_long = True
                if use_long:
                    if self.mode != 'minigrid_memory':
                        actions = actions.to(dtype=torch.long, device=states.device)
                        action_embeddings = self.action_embeddings(actions).squeeze(2) # (batch, block_size, n_embd)
                    elif self.mode == 'minigrid_memory':
                        actions = torch.where(actions == -10, torch.tensor(3), actions)
                        actions = actions.to(dtype=torch.long, device=states.device)
                    action_embeddings = self.action_embeddings(actions).squeeze(2) # (batch, block_size, n_embd)
                else:
                    action_embeddings = self.action_embeddings(actions) # (batch, block_size, n_embd)
                if self.mode == 'memory_maze':
                    if use_long:
                        action_embeddings = self.action_embeddings(actions).squeeze(1).squeeze(2)
                
                token_embeddings = torch.zeros((B, B1*3 - int(target is None), self.d_embed), dtype=torch.float32, device=state_embeddings.device)
                token_embeddings[:, ::3, :] = rtg_embeddings #+ time_embeddings
                token_embeddings[:, 1::3, :] = state_embeddings #+ time_embeddings
                token_embeddings[:, 2::3, :] = action_embeddings[:,-B1 + int(target is None):,:] #+ time_embeddings[:,-states.shape[1] + int(target is None):,:]
            else:
                token_embeddings = torch.zeros((B, B1*2, self.d_embed), dtype=torch.float32, device=state_embeddings.device)
                token_embeddings[:,::2,:] = rtg_embeddings #+ time_embeddings # really just [:,0,:]
                token_embeddings[:,1::2,:] = state_embeddings #+ time_embeddings # really just [:,1,:]

            hidden, new_mems = self._forward(token_embeddings, mems=mems, mem_tokens=mem_tokens)
            hidden = hidden.permute(1,0,2)

            num_mem = self.num_mem_tokens

            if self.num_mem_tokens > 0:
                if self.mem_at_end:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -num_mem:, :]
                else:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -tgt_len-num_mem:-tgt_len, :]

                if self.n_head_ca != 0:
                    # new_mem_tokens = F.relu(hidden[:, -num_mem:, :])
                    if self.mrv_act is not None:
                        new_mem_tokens = self.mrv_act(hidden[:, -num_mem:, :])
                    else:
                        new_mem_tokens = hidden[:, -num_mem:, :]

                    mem_tokens = mem_tokens.permute(1,0,2)
                    mask_mem_mem = torch.ones((new_mem_tokens.shape[1], new_mem_tokens.shape[1]), dtype=torch.bool).to(new_mem_tokens.device)
                    mem_tokens_write, _ = self.mha_mem_to_mem(mem_tokens, new_mem_tokens, new_mem_tokens, attn_mask=mask_mem_mem)

            if self.mem_at_end:
                logits = self.head(hidden)[:, num_mem:-num_mem]
            else:
                tgt_len = token_embeddings.shape[1]
                logits = self.head(hidden)[:, -tgt_len:] # was tgt_len

            if actions is not None:
                logits = logits[:, 1::3, :]
            else:
                logits = logits[:, 1:, :]
                
        #################################################### LOSS CALCULATION ######################################################################
        
        loss = None
        if target is not None:
            if self.mode == 'mujoco':
                loss = nn.MSELoss()(logits, target)
                
            if self.mode == 'tmaze':
                """ SELECT TARGETS FOR THE LAST LOSS """
                if self.flag == 1:
                    logits_last = torch.zeros((logits.shape[0], 1, 4))
                    target_last = torch.zeros((target.shape[0], 1, 1))
                    for batch_num in range(logits.shape[0]):
                        ind = torch.where(target[batch_num].squeeze()==-10)[0][0].item() - 1
                        logits_last[batch_num] = logits[batch_num, ind]
                        target_last[batch_num] = target[batch_num, ind]
                """ ================================ """
                
                """ CALCULATE TRAIN SUCCES RATE """
                if self.flag == 1:
                    train_sr = 0
                    with torch.no_grad():
                        for tr_batch_num in range(target.shape[0]):
                            y_real = target[tr_batch_num].squeeze()
                            mask_real = masks[tr_batch_num]
                            act_real = torch.sum(y_real * mask_real)
                            y_pred = torch.argmax(torch.softmax(logits[tr_batch_num].squeeze(), dim=-1), dim=-1)
                            act_pred = y_pred[torch.where(y_real != 0)[0][0].item()]
                            if act_pred == act_real:
                                train_sr += 1
                        self.last_acc = train_sr / target.shape[0]
                """ =========================== """
                
                # LOSSES
                ## ACCURACY
                probs = torch.softmax(logits, dim=-1)
                ans = torch.argmax(probs, dim=-1)
                for batch_num in range(target.shape[0]):
                    if -10 in target[batch_num]:
                        ind = torch.where(target[batch_num]==-10)[0][0].item()
                        ans[batch_num, ind:] = -10
                        
                labels = target.squeeze(-1)
                self.accuracy = torch.mean(torch.eq(ans, labels).float())
                
                ## LAST LOSS
                if self.flag == 1:
                    criterion_last = nn.CrossEntropyLoss(ignore_index=-10)
                    logits_last = logits_last.reshape(-1, logits_last.shape[-1])
                    target_last = target_last.reshape(-1).long()
                    self.loss_last = criterion_last(logits_last, target_last)
                
                ## FULL LOSS
                weight_not_important = 1
                weights_acts = torch.tensor([weight_not_important, 1, weight_not_important, 1], device=logits.device, dtype=torch.float32)
                criterion_all = nn.CrossEntropyLoss(ignore_index=-10, weight=weights_acts, reduction='mean')
                logits = logits.reshape(-1, logits.size(-1))
                target = target.reshape(-1).long()
                self.loss_all = criterion_all(logits, target)

            if self.mode == 'aar':
                """ START OF GIGALOSS """
                probs_0, probs_1 = 0, 0
                probs_0_cnt, probs_1_cnt = 0, 0
                acc_0, acc_1 = 0, 0

                logits_0 = torch.empty(1, 3).to(device=logits.device)
                targets_0 = torch.empty(1, 1).to(device=target.device)
                logits_1 = torch.empty(1, 3).to(device=logits.device)
                targets_1 = torch.empty(1, 1).to(device=target.device)

                # ? SINGLE
                for batch_num in range(target.shape[0]):
                    target_vector = target[batch_num]
                    logits_vector = logits[batch_num]
                    probs_vector = torch.softmax(logits_vector.squeeze(), dim=-1)
                    predicts_vector = torch.argmax(probs_vector, dim=-1)
                    if (target_vector == -10).all():
                        # padded vector
                        pass
                    else:
                        if (target_vector == 2).all():
                            # staying vector
                            pass
                        else:
                            # good for loss
                            ind = torch.where(target[batch_num]==-10)[0]
                            if len(ind) >= 1:
                                ind = ind[0]
                            if ind.numel() > 0:
                                ind = ind.item()
                                target_action = target_vector[ind-1].item()
                                if target_action == 0:
                                    prob_0 = probs_vector[ind-1][0].item()
                                    probs_0 += prob_0
                                    if predicts_vector[ind-1].item() == 0:
                                        acc_0 += 1
                                    probs_0_cnt += 1
                                    logits_0 = torch.concatenate([logits_0, logits_vector[ind-1].unsqueeze(0)])
                                    targets_0 = torch.concatenate([targets_0, target_vector[ind-1].unsqueeze(0)])
                                elif target_action == 1:
                                    prob_1 = probs_vector[ind-1][1].item()
                                    probs_1 += prob_1
                                    if predicts_vector[ind-1].item() == 1:
                                        acc_1 += 1
                                    probs_1_cnt += 1
                                    logits_1 = torch.concatenate([logits_1, logits_vector[ind-1].unsqueeze(0)])
                                    targets_1 = torch.concatenate([targets_1, target_vector[ind-1].unsqueeze(0)])

                probs_0_mean = probs_0 / probs_0_cnt if probs_0_cnt != 0 else 0
                probs_1_mean = probs_1 / probs_1_cnt if probs_1_cnt != 0 else 0

                acc_0 = acc_0 / probs_0_cnt if probs_0_cnt != 0 else 0
                acc_1 = acc_1 / probs_1_cnt if probs_1_cnt != 0 else 0

                logits_0, logits_1 = logits_0[1:], logits_1[1:]
                targets_0, targets_1 = targets_0[1:], targets_1[1:]

                self.loss_all_0, self.loss_all_1 = None, None
                
                if self.flag == 1:
                    criterion_0 = nn.CrossEntropyLoss(reduction='mean')
                    logits_0 = logits_0.reshape(-1, logits.size(-1))
                    targets_0 = targets_0.reshape(-1).long()
                    self.loss_all_0 = criterion_0(logits_0, targets_0)

                    criterion_1 = nn.CrossEntropyLoss(reduction='mean')
                    logits_1 = logits_1.reshape(-1, logits.size(-1))
                    targets_1 = targets_1.reshape(-1).long()
                    self.loss_all_1 = criterion_1(logits_1, targets_1)

                self.probs_0_mean, self.acc_0, self.loss_all_0 = probs_0_mean, acc_0, self.loss_all_0
                self.probs_1_mean, self.acc_1, self.loss_all_1 = probs_1_mean, acc_1, self.loss_all_1

                """ END OF GIGALOSS """
                
                """ SELECT TARGETS FOR THE LAST LOSS """
                if self.flag == 1:
                    logits_last = torch.zeros((logits.shape[0], 1, 3))
                    target_last = torch.zeros((target.shape[0], 1, 1))
                    for batch_num in range(logits.shape[0]):
                        ind = torch.where(target[batch_num]==-10)[0]
                        if len(ind) >= 1:
                            ind = ind[0] - 1
                        logits_last[batch_num] = logits[batch_num, ind]
                        target_last[batch_num] = target[batch_num, ind]
                """ ================================ """
                
                """ CALCULATE TRAIN SUCCES RATE """
                if self.flag == 1:
                        self.last_acc = (self.acc_0 + self.acc_1) / 2
                """ =========================== """
                
                # LOSSES
                ## ACCURACY
                probs = torch.softmax(logits, dim=-1)
                ans = torch.argmax(probs, dim=-1)
                for batch_num in range(target.shape[0]):
                    if -10 in target[batch_num]:
                        ind = torch.where(target[batch_num]==-10)[0][0].item()
                        ans[batch_num, ind:] = -10
                        
                labels = target.squeeze(-1)
                self.accuracy = torch.mean(torch.eq(ans, labels).float())
                
                ## LAST LOSS
                if self.flag == 1:
                    criterion_last = nn.CrossEntropyLoss(ignore_index=-10)
                    logits_last = logits_last.reshape(-1, logits_last.shape[-1])
                    target_last = target_last.reshape(-1).long()
                    self.loss_last = criterion_last(logits_last, target_last)
                
                ## FULL LOSS
                criterion_all = nn.CrossEntropyLoss(ignore_index=-10, reduction='mean')
                logits = logits.reshape(-1, logits.size(-1))
                target = target.reshape(-1).long()
                self.loss_all = criterion_all(logits, target)
            
            
            if self.mode == 'associative-retrieval-seq-to-num':
                self.accuracy = 0
                self.loss_all = torch.tensor(0)
                self.loss_last = torch.tensor(0)
                
            if self.mode == 'doom':
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       target.reshape(-1).long())

            if self.mode == 'memory_maze':
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       target.reshape(-1).long())
                
            if self.mode == 'minigrid_memory':
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       target.reshape(-1).long(), ignore_index=-10)
            
            if self.mode == 'mujoco':
                # loss = None
                loss_fn = nn.MSELoss()
                if target is not None:
                    loss = loss_fn(logits, target)
            if self.mode == 'atari':
                if target is not None:
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1).long())
            if self.mode == 'key_to_door':
                masks = masks[:, :].unsqueeze(-1)
                target = target * masks
                logits = logits * masks.expand(-1, -1, logits.size(-1))
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1).long())    
                
        
        output = [logits, loss]
        if new_mems is not None:
            output = output + new_mems
        
        if self.num_mem_tokens != 0:
            output = output, mem_tokens_write.permute(1,0,2)
        else:
            output = output, None
        
        return output

######################################################################################    

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, scale=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.scale = scale

    def forward(self, query, key, value, attn_mask=None):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / query.size(-1)**0.5 if self.scale is None else self.scale
    
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        if self.is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        #attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout_p, train=True)
        output = torch.matmul(attn_weight, value)
        return output, attn_weight
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_model, num_heads, dropout_p=0.0, is_causal=False, scale=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_p, is_causal, scale)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_q, d_model)
        self.wk = nn.Linear(d_k, d_model)
        self.wv = nn.Linear(d_v, d_model)
        
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)
        # q: (bs, sl, d_q)
        # k: (bs, sl, d_k)
        # q: (bs, sel, d_v)
        
        q = self.wq(q) # (bs, sl, d_model)
        k = self.wk(k) # (bs, sl, d_model)
        v = self.wv(v) # (bs, sl, d_model)
        
        q = self.split_heads(q, batch_size) # (bs, num_heads, seq_len, depth=d_model/num_heads)
        k = self.split_heads(k, batch_size) # (bs, num_heads, seq_len, depth=d_model/num_heads)
        v = self.split_heads(v, batch_size) # (bs, num_heads, seq_len, depth=d_model/num_heads)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, attn_mask)
        
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        
        # (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        
        return output, attention_weights
