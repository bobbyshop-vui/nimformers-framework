"""``torch.nn.init``: no-ops, parameter values always come from the checkpoint."""


def _keep(tensor, *args, **kwargs): return tensor


normal_ = uniform_ = zeros_ = ones_ = constant_ = eye_ = dirac_ = orthogonal_ = sparse_ = _keep
xavier_uniform_ = xavier_normal_ = kaiming_uniform_ = kaiming_normal_ = trunc_normal_ = _keep
normal = uniform = constant = xavier_uniform = xavier_normal = kaiming_uniform = kaiming_normal = _keep


def calculate_gain(nonlinearity, param=None): return 1.0
