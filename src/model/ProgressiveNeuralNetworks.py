import torch
import torch.nn as nn
import torch.nn.functional as F

class PNNLinearBlock(nn.Module):
    def __init__(self, in_sizes, out_size, scalar_mult=1.0):
        super(PNNLinearBlock, self).__init__()
        assert isinstance(in_sizes, (list, tuple))
        self.in_sizes = in_sizes
        self.w = nn.Linear(in_sizes[-1], out_size)

        self.v = nn.ModuleList()
        self.alphas = nn.ParameterList()
        for in_size in in_sizes[:-1]:
            new_alpha = torch.tensor([scalar_mult])#.expand(in_size)
            self.alphas.append(nn.Parameter(new_alpha))
            self.v.append(nn.Linear(in_size, out_size))
        if len(in_sizes) > 1:
            self.u = nn.Linear(out_size, out_size)
        else:
            self.u = None

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        cur_column_out = self.w(inputs[-1])

        # prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]
        prev_columns_out = []
        for x, alpha, v in zip(inputs, self.alphas, self.v):
            prev_columns_out.append(v(alpha * x))
        prev_columns_out = sum(prev_columns_out)
        if self.u:
            prev_columns_out = self.u(F.relu(prev_columns_out))
        return cur_column_out + prev_columns_out


class PNN(nn.Module):
    def __init__(self, n_layers):
        super(PNN, self).__init__()
        self.n_layers = n_layers
        self.columns = nn.ModuleList([])

        self.use_cuda = False

    def forward(self, x, task_id=-1):
        assert self.columns, 'PNN should at least have one column (missing call to `new_task` ?)'
        inputs = [c[0](x) for c in self.columns]

        for l in range(1, self.n_layers):
            inputs = list(map(F.relu, inputs))
            outputs = []
            #TODO: Use task_id to check if all columns are necessary
            for i, column in enumerate(self.columns):
                outputs.append(column[l](inputs[:i+1]))

            inputs = outputs

        return inputs[task_id]

    def new_task(self, sizes):
        msg = "Should have the out size for each layer + input size (got {} sizes but {} layers)."
        assert len(sizes) == self.n_layers + 1, msg.format(len(sizes), self.n_layers)
        task_id = len(self.columns)

        modules = []
        modules.append(PNNLinearBlock([sizes[0]], sizes[1]))
        for i in range(1, self.n_layers):
            new_block = PNNLinearBlock([sizes[i]]*(task_id + 1), sizes[i+1])
            modules.append(new_block)
        new_column = nn.ModuleList(modules)
        self.columns.append(new_column)

        if self.use_cuda:
            self.cuda()

    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

    def parameters(self, col=None):
        if col is None:
            return super(PNN, self).parameters()
        return self.columns[col].parameters()

    def cuda(self, *args, **kwargs):
        self.use_cuda = True
        super(PNN, self).cuda(*args, **kwargs)

    def cpu(self):
        self.use_cuda = False
        super(PNN, self).cpu()

