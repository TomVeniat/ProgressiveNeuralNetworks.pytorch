import torch
import torch.nn as nn
import torch.nn.functional as F


class PNNLinearBlock(nn.Module):
    def __init__(self, col, depth, n_in, n_out):
        super(PNNLinearBlock, self).__init__()
        self.col = col
        self.depth = depth
        self.n_in = n_in
        self.n_out = n_out
        self.w = nn.Linear(n_in, n_out)

        self.u = nn.ModuleList()
        if self.depth > 0:
            self.u.extend([nn.Linear(n_in, n_out) for _ in range(col)])

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        cur_column_out = self.w(inputs[-1])
        prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]

        return F.relu(cur_column_out + sum(prev_columns_out))


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
        for i in range(0, self.n_layers):
            modules.append(PNNLinearBlock(task_id, i, sizes[i], sizes[i+1]))
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

