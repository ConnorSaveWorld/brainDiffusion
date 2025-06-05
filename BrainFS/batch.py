import torch
import torch_geometric
from torch_geometric.data import Data
import pdb

# This is a copy from torch_geometric/data/batch.py
# which is modified to support batch asignment in subgraph level

# noinspection PyPackageRequirements
class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys()) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        if 'assignment_index_2' in keys:
            cumsum['assignment_index_2'] = torch.LongTensor([[0], [0]])
        if 'assignment_index_3' in keys:
            cumsum['assignment_index_3'] = torch.LongTensor([[0], [0]])

        sc_batch_list = []
        fc_batch_list = []

        cumsum = {key: 0 for key in keys}
        for i, data in enumerate(data_list):
            # --- Keep track of num_nodes per modality ---
            current_sc_nodes = data.sc_num_nodes if 'sc_num_nodes' in data else 0
            current_fc_nodes = data.fc_num_nodes if 'fc_num_nodes' in data else 0

            for key in data.keys():
                item = data[key]
                # --- Apply increments correctly ---
                # (Using the simplified logic from previous fix - assumes __inc__ handles it)
                if key in ('sc_edge_index', 'fc_edge_index') and torch.is_tensor(item):
                    # This assumes data.__inc__(key, item) is correctly implemented in Batch
                    # or the standard Data class based on sc_num_nodes / fc_num_nodes
                    item = item + cumsum[key]
                    cumsum[key] = cumsum[key] + data.__inc__(key, item)
                elif torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                    # Increment other keys if needed based on __inc__
                    if key not in ('sc_edge_index', 'fc_edge_index'):
                        cumsum[key] = cumsum[key] + data.__inc__(key, item)

                # --- Size calculation and slicing (remains same) ---
                if torch.is_tensor(item):
                    current_item_for_size = item
                    size = current_item_for_size.size(data.__cat_dim__(key, current_item_for_size))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                batch[key].append(item)  # Append potentially incremented item

                # --- Original follow_batch logic ---
                if key in follow_batch:
                    follow_item = torch.full((size,), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(follow_item)

            # --- Create MODALITY-SPECIFIC batch tensors ---
            if current_sc_nodes > 0:
                sc_item = torch.full((current_sc_nodes,), i, dtype=torch.long)
                sc_batch_list.append(sc_item)
            if current_fc_nodes > 0:
                fc_item = torch.full((current_fc_nodes,), i, dtype=torch.long)
                fc_batch_list.append(fc_item)

            # --- Final Concatenation ---
        for key in batch.keys():
            if batch[key]:  # Check if list is not empty
                item = batch[key][0]
                if torch.is_tensor(item):
                    batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, item))
                elif isinstance(item, int) or isinstance(item, float):
                    batch[key] = torch.tensor(batch[key])
            # else: # Keep key with empty list or handle as None/delete
            #    batch[key] = None

            # --- Concatenate the modality-specific batch lists ---
        if sc_batch_list:
            batch.sc_batch = torch.cat(sc_batch_list, dim=0)
        else:
            batch.sc_batch = None  # Or an empty tensor

        if fc_batch_list:
            batch.fc_batch = torch.cat(fc_batch_list, dim=0)
        else:
            batch.fc_batch = None

            # --- SAFER APPROACH: Omit top-level 'batch' if structure is sc_/fc_ based ---
            # If your model accesses nodes via sc_x, fc_x etc., you might not need
            # a top-level 'batch' attribute mapping combined nodes.
            # Commenting this out for now, reinstate carefully if needed.
            # num_nodes = data.num_nodes # THIS IS LIKELY WRONG/NONE
            # if num_nodes is not None:
            #     item = torch.full((num_nodes, ), i, dtype=torch.long)
            #     batch.batch.append(item)

        # Copy custom data functions to batch (does not work yet):
        # if data_list.__class__ != Data:
        #     org_funcs = set(Data.__dict__.keys())
        #     funcs = set(data_list[0].__class__.__dict__.keys())
        #     batch.__custom_funcs__ = funcs.difference(org_funcs)
        #     for func in funcs.difference(org_funcs):
        #         setattr(batch, func, getattr(data_list[0], func))

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using Batch.from_data_list()'))

        keys = [key for key in self.keys if key[-5:] != 'batch']
        cumsum = {key: 0 for key in keys}
        if 'assignment_index_2' in keys:
            cumsum['assignment_index_2'] = torch.LongTensor([[0], [0]])
        if 'assignment_index_3' in keys:
            cumsum['assignment_index_3'] = torch.LongTensor([[0], [0]])
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                if torch.is_tensor(self[key]):
                    data[key] = self[key].narrow(
                        data.__cat_dim__(key,
                                         self[key]), self.__slices__[key][i],
                        self.__slices__[key][i + 1] - self.__slices__[key][i])
                    if self[key].dtype != torch.bool:
                        data[key] = data[key] - cumsum[key]
                else:
                    data[key] = self[key][self.__slices__[key][i]:self.
                                          __slices__[key][i + 1]]
                if key == 'node_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'subgraph_to_graph':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'original_edge_index':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'tree_edge_index':
                    cumsum[key] = cumsum[key] + data.num_cliques
                elif key == 'atom2clique_index':
                    cumsum[key] = cumsum[key] + torch.tensor([[data.num_atoms], [data.num_cliques]])
                elif key == 'edge_index_2':
                    cumsum[key] = cumsum[key] + data.iso_type_2.shape[0]
                elif key == 'edge_index_3':
                    cumsum[key] = cumsum[key] + data.iso_type_3.shape[0]
                elif key == 'batch_2':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'batch_3':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'assignment2_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment3_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment_index_2':
                    cumsum[key] = cumsum[key] + torch.LongTensor([[data.num_nodes], [data.iso_type_2.shape[0]]])
                elif key == 'assignment_index_3':
                    cumsum[key] = cumsum[key] + torch.LongTensor([[data.iso_type_2.shape[0]], [data.iso_type_3.shape[0]]])
                else:
                    cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Define how different attributes should be concatenated
        # Default PyG behavior is usually fine, but check for prefixed keys
        if key in ('sc_edge_index', 'fc_edge_index', 'edge_index'):
            return 1  # Concatenate edge indices along the second dimension
        elif key in ('sc_edge_attr', 'fc_edge_attr', 'edge_attr'):
            return 0  # Concatenate edge attributes along the first dimension
        elif key in ('sc_x', 'fc_x', 'x'):
            return 0  # Concatenate node features along the first dimension
        elif key == 'y':
            return 0  # Concatenate labels along the first dimension
        # Add other custom keys if needed
        else:
            # Fallback to default Data behavior
            return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # Define how indices should be incremented for specific keys when batching
        # This is crucial for edge_index attributes
        if key == 'sc_edge_index':
            # Increment by the number of nodes in the SC graph of the *previous* item
            # This requires access to previous item's sc_num_nodes, handled in from_data_list
            return self[f'sc_num_nodes']  # Return the increment value needed
        elif key == 'fc_edge_index':
            # Increment by the number of nodes in the FC graph of the *previous* item
            return self[f'fc_num_nodes']
        # Add other custom keys if needed
        else:
            # Fallback to default Data behavior (usually 0 for most attributes)
            return super().__inc__(key, value, *args, **kwargs)