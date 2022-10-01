import torch
import torch.nn as nn

__all__ = ['TransformerEncoder']


class TransformerEncoder(nn.Module):
    """
    Input of a single modality, which is encoded
    """

    def __init__(self, hidden_size: int, num_layers: int, nhead: int, in_size: int, feature_size: int,
                 input_len: int = None, output_len: int = None, return_cls: bool = False, no_attention: bool = False,
                 freeze_adapt: bool = False, freeze_all: bool = False, use_time: bool = True,
                 no_attention_test: bool = False, return_attention=False, dropout: float = 0.1,
                 treat_zero_padding: bool = True, num_latent_dims: int = 1, precompute_time_embs: bool = True,
                 size_out: int = None):
        super().__init__()

        self.in_size = in_size
        self.size_out = feature_size * num_latent_dims if size_out is None else size_out
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hidden_size
        self.return_cls = return_cls
        self.no_attention = no_attention
        self.use_time = use_time
        self.nhead = nhead
        # Similar to no_attention, but used for testing models that have been trained with attention. Therefore, we
        # still use temporal embeddings (even though it will not be useful), so that it is not out of distribution
        self.no_attention_test = no_attention_test
        self.return_attention = return_attention
        self.dropout = dropout
        self.treat_zero_padding = treat_zero_padding
        self.precompute_time_embs = precompute_time_embs

        if self.return_attention:
            encoder_layer = TransformerEncoderLayerReturnAttention(d_model=hidden_size, nhead=nhead, dropout=dropout)
            self.encoder = TransformerEncoderReturnAttention(encoder_layer=encoder_layer, num_layers=num_layers)
            self.discard_ratio = 0.9
            # self.head_fusion = 'max'
            self.head_fusion = "already_fused"
            for name, module in self.named_modules():
                if name.endswith('self_attn'):
                    module.register_forward_hook(self.get_attention)
                    module.register_full_backward_hook(self.get_attention_gradient)
            self.attention_gradients = []
            self.attentions = []

        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        self.adapt_in = nn.Linear(in_size, hidden_size)
        self.adapt_out = nn.Linear(hidden_size, self.size_out)

        self.special_tokens = nn.Embedding(2, hidden_size)  # CLS, to_predict

        def special_token(idx, d):
            idx_tensor = torch.tensor(idx).to(self.special_tokens.weight.device)
            token = self.special_tokens(idx_tensor)
            token = token[None, None, :].expand(1, d, self.hidden_size)
            return token

        self.predict_token = lambda d=1: special_token(1, d)

        self.cls_exists = False
        if self.return_cls:
            self.cls_token = lambda d=1: special_token(0, d)
            self.cls_exists = True

        if (not no_attention) and use_time and not precompute_time_embs:
            self.temporal_embeddings = nn.Embedding(input_len + output_len, hidden_size)

        if freeze_adapt:
            for parameter in self.adapt_in.parameters():
                parameter.requires_grad = False

        if freeze_all:
            for name, parameter in self.named_parameters():
                parameter.requires_grad = False

    def get_attention(self, module, input, output):
        if input[0].requires_grad:
            self.attentions.append(output[1].cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[2].cpu())

    def forward_attention(self, x, x_len=None, target=None):

        self.attentions = []
        self.attention_gradients = []
        self.zero_grad()
        output = self.forward(x, x_len)
        rollout_out = rollout(self.attentions, self.discard_ratio, self.head_fusion)
        grad_rollout_out = None

        return rollout_out, grad_rollout_out

    def forward(self, x, x_len=None, temporal_embeddings=None):
        """
        :param x: size [B, T, C]
        :param x_len: [B]
        :param temporal_embeddings: [B, T, H]
        """
        x = self.adapt_in(x.permute(1, 0, 2))  # T, B, hidden_dim

        # Concatenate to_predict tokens
        if self.output_len is not None:
            x = torch.cat([x, self.predict_token(self.output_len).repeat(x.shape[1], 1, 1).transpose(1, 0)])

        if (not self.no_attention) and self.use_time:
            if temporal_embeddings is None:
                assert not self.precompute_time_embs, 'Make sure time_indices_embeddings is True to create Embeddings'
                time_indices = torch.arange(x.shape[0]).to(x.device)
                temporal_embeddings = self.temporal_embeddings(time_indices).unsqueeze(1)
            else:
                temporal_embeddings = temporal_embeddings.permute(1, 0, 2)
            x += temporal_embeddings

        if self.return_cls:
            x = torch.cat([self.cls_token(x.shape[1]), x], dim=0)

        scale = torch.tensor(-1e4)
        if self.no_attention or self.no_attention_test:
            mask_pos = torch.eye(x.shape[0])
            mask = scale * (1 - mask_pos).to(x.device)
            # mask = ~mask_pos.bool().to(x.device)
        elif x_len is None:
            mask_pos = None
            mask = None
        else:
            mask = None
            if self.treat_zero_padding:
                # Avoid paying attention to zero-padding
                mask_pos = torch.zeros((x.shape[1], x.shape[0], x.shape[0]))
                # Pretty sure there's a matrix way of doing this, but this is very fast
                for i in range(x.shape[1]):
                    len_mask = x_len[i] + (1 if self.cls_exists else 0)
                    mask_pos[i, :len_mask, :len_mask] = 1
                # Expand for num_heads
                mask_pos = mask_pos.unsqueeze(1).expand(mask_pos.shape[0], self.nhead, mask_pos.shape[1],
                                                        mask_pos.shape[2])
                mask_pos = mask_pos.reshape(x.shape[1] * self.nhead, x.shape[0], x.shape[0])
                mask = scale * (1 - mask_pos).to(x.device)
                # mask = ~mask_pos.bool().to(x.device)

        out = self.encoder(x, mask=mask)

        # Return to [B, T, feat_dim/ order
        out = out.permute(1, 0, 2)

        if self.return_cls:
            out_cls = out[:, 0]
            out_seq = out[:, 1:]
            out = out_cls
        else:
            start = 0 if self.output_len is None else -self.output_len
            out = out[:, start:]

        out = self.adapt_out(out)
        return out


class TransformerEncoderLayerReturnAttention(nn.TransformerEncoderLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attention = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention


class TransformerEncoderReturnAttention(nn.TransformerEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, src, mask=None, src_key_padding_mask=None):

        output = src

        for mod in self.layers:
            output, attention = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention * weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            # indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result  # [0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    # width = int(mask.size(-1) ** 0.5)
    # mask = mask.reshape(width, width).numpy()
    mask = mask / torch.max(mask)
    return mask


def rollout(attentions, discard_ratio, head_fusion):
    discard_ratio = 0.
    result = torch.eye(attentions[0].size(-1)).unsqueeze(0).expand(*attentions[0].shape)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "already_fused":
                attention_heads_fused = attention
            elif head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused  # .view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I.unsqueeze(0)) / 2
            a = a / a.sum(dim=-1, keepdims=True)

            result = torch.bmm(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    # mask = result[0, 0, 1:]
    mask = result
    # width = int(mask.size(-1) ** 0.5)
    # mask = mask.reshape(width, width).numpy()
    mask = mask / torch.max(mask)
    return mask


import torch.nn.functional
from torch.nn.functional import _in_projection, linear
from torch.nn.functional import _pad as pad
from torch.overrides import (
    has_torch_function, handle_torch_function)


def new_multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight,
        out_proj_bias,
        training: bool = True,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        use_separate_proj_weight: bool = False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = torch.nn.functional._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            # warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        # warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    # attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output, attn_output_weights = my_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    if attn_output.isnan().any():
        print('hey')
        my_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.max(dim=1)[0]  # / num_heads  # <<<< CHANGE!
    else:
        return attn_output, None


@torch.autocast(dtype=torch.float32, device_type="cuda")
def my_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p):
    import math
    B, Nt, E = q.shape
    q2 = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q2, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q2, k.transpose(-2, -1))

    attn = torch.nn.functional.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


torch.nn.functional.multi_head_attention_forward = new_multi_head_attention_forward
