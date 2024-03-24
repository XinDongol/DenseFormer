import torch


class InPlaceSetSlice(torch.autograd.Function):
  """
  This function allows us to set value given an index in a tensor while the gradient
  can still be calculated.
  """

  @staticmethod
  def forward(ctx, full_tensor, last_slice, x_idx, x_val):
    full_tensor[x_idx] = x_val
    ctx.x_idx = x_idx
    ret = torch.Tensor().to(full_tensor.device)
    ret.set_(full_tensor[:x_idx + 1])
    return ret

  @staticmethod
  def backward(ctx, grad_out):
    if ctx.x_idx == 0:
      return None, None, None, grad_out[ctx.x_idx]
    else:
      return None, grad_out[:ctx.x_idx], None, grad_out[ctx.x_idx]


def apply_inplace_set(x_acc, x_idx, x_val):
  full_tensor, last_slice = x_acc
  new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val)  # last_slice is not used. 
  return full_tensor, new_slice # generally, the new slice is the modified full_tensor[:x_idx + 1]


class DWAModules(torch.nn.Module):

  def __init__(self, n_blocks, dilation=1, period=1):
    super().__init__()
    self.n_blocks = n_blocks
    self.dilation = dilation
    self.period = period
    self.alphas = torch.nn.ModuleList([torch.nn.Linear((i+1+dilation)//dilation, 1, bias=False) if (i+1)%period == 0 else None for i in range(n_blocks)])
    self.accumulators = None
    self._init_weights()

    # assert 1==0

  def _init_weights(self):
    for module in self.alphas:
      if module is not None:
        module.weight.data.zero_()
        module.weight.data[0, -1] = 1.

  def init_accumulators(self, x):
    """
    Let's assume self.dilation = 1 and self.period = 1

    self.accumulators = x_accs is a list of tuples 
    Its size is 
    [
      self.dilation * 
        (
          group_size * x.shape,   # saved blocks' output
          None                    # saved block's output that will be accumulated
        )
    ]
    """
    x_accs = []
    for i in range(self.dilation):
      current_group_size = (self.n_blocks + 1) // self.dilation
      if i < (self.n_blocks + 1) % self.dilation:
        current_group_size += 1
      x_accs.append(
        (
          torch.zeros((current_group_size, *x.shape), device=x.device, dtype=x.dtype), 
          None
        )
      )
    x_accs[0] = apply_inplace_set(
      x_accs[0], 
      0,  # set the first tensor (out of group_size) to x, which is the input tensor
      x
    )  
    self.accumulators = x_accs

  def forward(self, x, block_idx):
    # breakpoint()
    assert self.accumulators is not None, "`init_accumulators(x)` needs to be called first"

    # save the output in order to use them later
    self.accumulators[(block_idx+1) % self.dilation] = apply_inplace_set(
        self.accumulators[(block_idx+1) % self.dilation], # if self.dilation = 1, this idx is always 0
        (block_idx+1)//self.dilation,                     # if self.dilation = 1, this idx is always (block_idx + 1), which means that we save the output of the block_idx
        x  
    )

    # breakpoint()

    # use the saved output from previous block
    if (block_idx+1) % self.period == 0:
      x = torch.tensordot(self.alphas[block_idx].weight.view(-1), self.accumulators[(block_idx+1)%self.dilation][1], dims=1)
      # breakpoint()
      # if self.dilation = 1, 
      # the second term is ( group_size * x.shape, None )[1]
    return x
