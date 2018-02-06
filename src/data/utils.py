import torch


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length, transform):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        self.transform = transform
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        self.parent_ds.transform = self.transform
        return self.parent_ds[i + self.offset]


def validation_split(dataset, train_transforms, val_transforms, val_size=None, val_share=0.1):
    """
       Split a (training and validation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

       """

    val_offset = len(dataset) - val_size if val_size is not None else int(len(dataset) * (1 - val_share))
    assert val_offset > 0, "Can't extract a size {} validation set out of a size {} dataset".format(val_size, len(dataset))
    return PartialDataset(dataset, 0, val_offset, train_transforms), PartialDataset(dataset, val_offset, len(dataset) - val_offset, val_transforms)
