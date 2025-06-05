import os
import pickle
import numpy as np
import torch
import scipy.io as sio # Use scipy.io for .mat files
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from sklearn.preprocessing import normalize
# from batch import Batch # Assuming batch.py contains the Batch class
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_single_modality_data(adj, feature):
    """ Creates PyG Data for a single modality """
    # Ensure inputs are numpy arrays first if needed
    if not isinstance(adj, np.ndarray):
        adj = np.array(adj)
    if not isinstance(feature, np.ndarray):
        feature = np.array(feature)

    # Ensure correct dtype
    adj = adj.astype(np.float32)
    feature = feature.astype(np.float32)

    adj_tensor = torch.from_numpy(adj)
    edge_index, edge_weight = dense_to_sparse(adj_tensor)
    # Use the feature matrix directly as node features 'x' for the transform
    x = torch.from_numpy(feature)
    # Store the original adjacency matrix if needed by transform or later use
    original_adj = torch.from_numpy(adj)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, num_nodes=adj.shape[0])
    # Optionally add original_adj if needed, though HHopSubgraphs uses edge_index primarily
    # data.original_adj = original_adj
    return data

class SubgraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'sc_edge_index':
            # Increment based on the number of SC subgraph nodes in the *current* data object being added
            # This requires knowing how many nodes are in sc_x (subgraph nodes) for *this* specific sample
            # We can get this from sc_node_to_subgraph's size for this sample
            if hasattr(self, 'sc_x') and self.sc_x is not None:
                 return self.sc_x.size(0) # Number of nodes in the subgraphs for this sample
            else: return 0 # Should not happen if data is valid
        elif key == 'fc_edge_index':
            if hasattr(self, 'fc_x') and self.fc_x is not None:
                return self.fc_x.size(0) # Number of nodes in the subgraphs for this sample
            else: return 0
        elif key == 'sc_node_to_subgraph':
            # Increment by the number of SC subgraphs in the *current* data object
            # This is the max value + 1, or simply the size of sc_subgraph_to_graph
            if hasattr(self, 'sc_subgraph_to_graph') and self.sc_subgraph_to_graph is not None:
                return self.sc_subgraph_to_graph.size(0) # Number of subgraphs for this sample
            else: return 0
        elif key == 'fc_node_to_subgraph':
            if hasattr(self, 'fc_subgraph_to_graph') and self.fc_subgraph_to_graph is not None:
                return self.fc_subgraph_to_graph.size(0) # Number of subgraphs for this sample
            else: return 0
        elif key == 'sc_original_edge_index':
             # Increment by the number of original nodes
             if hasattr(self, 'sc_original_x') and self.sc_original_x is not None:
                 return self.sc_original_x.size(0)
             else: return 0
        elif key == 'fc_original_edge_index':
             # Increment by the number of original nodes
             if hasattr(self, 'fc_original_x') and self.fc_original_x is not None:
                 return self.fc_original_x.size(0)
             else: return 0
        # Add other keys specific to your HHopSubgraphs output if needed
        # For example, if LapEncoding adds 'assignment_index_2', handle that too.
        # elif key == 'assignment_index_2': # Example from transform.py
        #     return torch.tensor([[data.num_atoms], [data.num_cliques]]) # Adjust based on actual attributes
        else:
            # Fallback to the default increment behavior
            return super().__inc__(key, value, *args, **kwargs)

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, args=None):
        self.args = args

        # self.raw_data_root = '/home/jinghu/diffusion/Graph-U-Nets-new/data'  # Root of the .mat files
        # self.sc_mat_filename = 'scnHCPD.mat'
        # self.fc_mat_filename = 'fcnHCPD.mat'
        # self.label_mat_filename = 'labHCPD.mat'
        # self.sc_mat_key = 'scn_corr'
        # self.fc_mat_key = 'fcn_corr'


        # self.raw_data_root = '/home/jinghu/diffusion/testDiff/BrainSTORE-3CE9/GraphExp/datasets/ppmi'  # Root of the .mat files
        # self.sc_mat_filename = 'scn_corrHcPd/processed_second_partition.mat'
        # self.fc_mat_filename = 'fcn_corrHcPd/processed_second_partition.mat'
        # self.label_mat_filename = 'labHCPD.mat'
        # self.sc_mat_key = 'processed_matrices'
        # self.fc_mat_key = 'processed_matrices'

        
        # self.label_mat_key = 'labels'

        self.raw_data_root = '/home/jinghu/diffusion/Graph-U-Nets-new/data'
        self.sc_pkl_filename =  'sc_site_16.pkl'
        self.fc_pkl_filename =  'fc_site_16.pkl'       
        self.label_pkl_filename =  'site_16_labels.csv'


        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        # Point to the directory containing the raw .mat files
        return self.raw_data_root

    @property
    def raw_file_names(self):
        return [self.sc_mat_filename, self.fc_mat_filename, self.label_mat_filename]

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...
    def safe_load_pickle(file_obj):
        try:
        # Try to load pickle directly
            return pickle.load(file_obj)
        except Exception as e1:
            print(f"Standard pickle load failed: {e1}")
            try:
            # Try with encoding for Python 2/3 compatibility
                file_obj.seek(0)  # Reset file pointer
                return pickle.load(file_obj, encoding='latin1')
            except Exception as e2:
                print(f"Alternative pickle load failed: {e2}")
            # As a last resort, try to load raw bytes and process manually
                file_obj.seek(0)
                raw_data = file_obj.read()
            # This will depend on what's in your pickle files
            # You might need a custom parser if this fails
                return {'data': raw_data}

    def process(self):
        print(f"--- Starting Dataset Processing ---")
        print(f"Processing data from raw directory: {self.raw_dir}")

        
        # sc_path = os.path.join(self.raw_dir, self.sc_mat_filename)
        # fc_path = os.path.join(self.raw_dir, self.fc_mat_filename)
        # label_path = os.path.join(self.raw_dir, self.label_mat_filename)

        # # --- Load Raw Data (Keep error handling as before) ---
        # try:
        #     sc_data = sio.loadmat(sc_path)
        #     fc_data = sio.loadmat(fc_path)
        #     label_data = sio.loadmat(label_path)
        #     if self.label_mat_key not in label_data: raise KeyError(f"Key '{self.label_mat_key}' not found in label file: {label_path}")
        #     all_labels = label_data[self.label_mat_key].flatten()
        #     if self.fc_mat_key not in fc_data: raise KeyError(f"Key '{self.fc_mat_key}' not found in FC file: {fc_path}")
        #     all_fc_matrices_obj = fc_data[self.fc_mat_key]
        #     if self.sc_mat_key not in sc_data: raise KeyError(f"Key '{self.sc_mat_key}' not found in SC file: {sc_path}")
        #     all_sc_matrices_obj = sc_data[self.sc_mat_key]
        #     print("Raw .mat files loaded successfully.")
        # except Exception as e:
        #     print(f"FATAL ERROR during file loading: {e}")
        #     raise e

        # # --- Validate Data (Keep as before) ---
        # # Handle potential differences in how data is stored (direct array vs object array)
        # if isinstance(all_sc_matrices_obj, np.ndarray) and all_sc_matrices_obj.dtype == 'object':
        #     num_subjects_sc = all_sc_matrices_obj.shape[0]
        #     is_sc_obj_array = True
        # elif isinstance(all_sc_matrices_obj, np.ndarray) and all_sc_matrices_obj.ndim == 3: # Check if it's N x Nodes x Nodes
        #      num_subjects_sc = all_sc_matrices_obj.shape[0]
        #      is_sc_obj_array = False
        # else:
        #     raise TypeError(f"Unexpected SC data format: {type(all_sc_matrices_obj)}, shape: {getattr(all_sc_matrices_obj, 'shape', 'N/A')}")

        # if isinstance(all_fc_matrices_obj, np.ndarray) and all_fc_matrices_obj.dtype == 'object':
        #     num_subjects_fc = all_fc_matrices_obj.shape[0]
        #     is_fc_obj_array = True
        # elif isinstance(all_fc_matrices_obj, np.ndarray) and all_fc_matrices_obj.ndim == 3: # Check if it's N x Nodes x Nodes
        #      num_subjects_fc = all_fc_matrices_obj.shape[0]
        #      is_fc_obj_array = False
        # else:
        #     raise TypeError(f"Unexpected FC data format: {type(all_fc_matrices_obj)}, shape: {getattr(all_fc_matrices_obj, 'shape', 'N/A')}")
        # num_labels = all_labels.shape[0]
        # if not (num_labels == num_subjects_fc == num_subjects_sc):
        #     msg = f"FATAL ERROR: Mismatch in subject counts! Labels:{num_labels}, FC:{num_subjects_fc}, SC:{num_subjects_sc}"
        #     print(msg); raise ValueError(msg)
        # n_subjects = num_labels
        # print(f"Found data for {n_subjects} subjects.")

        # # --- Create Initial Data Lists (Separate Modalities) ---
        # data_sc_list = []
        # data_fc_list = []
        # y_list = []
        # skipped_subjects = 0
        # for i in tqdm(range(n_subjects), desc="Reading Subjects & Creating Initial Data"):
        #      # (Keep the loop for reading/validating/normalizing SC/FC adj/features as before)
        #     subj_sc_adj = None; subj_fc_adj = None; subj_sc_feature = None; subj_fc_feature = None
        #     try:
        #         label = torch.LongTensor([int(all_labels[i])])
        #         current_sc_data = all_sc_matrices_obj[i][0] if is_sc_obj_array else all_sc_matrices_obj[i]
        #         if not isinstance(current_sc_data, np.ndarray) or current_sc_data.size == 0: print(f"W: S{i}, skip SC"); skipped_subjects += 1; continue
        #         subj_sc_adj = current_sc_data; subj_sc_feature = subj_sc_adj.copy()
        #         current_fc_data = all_fc_matrices_obj[i][0] if is_fc_obj_array else all_fc_matrices_obj[i]
        #         if not isinstance(current_fc_data, np.ndarray) or current_fc_data.size == 0: print(f"W: S{i}, skip FC"); skipped_subjects += 1; continue
        #         subj_fc_adj = current_fc_data; subj_fc_feature = subj_fc_adj.copy()
        #     except Exception as e: print(f"E: S{i} matrix: {e}. Skip."); skipped_subjects += 1; continue
        #     if subj_sc_adj.ndim != 2 or subj_fc_adj.ndim != 2 or \
        #        subj_sc_adj.shape[0] != subj_sc_adj.shape[1] or subj_fc_adj.shape[0] != subj_fc_adj.shape[1] or \
        #        subj_sc_adj.shape != subj_fc_adj.shape: print(f"W: S{i} matrix shape. Skip."); skipped_subjects += 1; continue
        #     if subj_sc_adj.shape[0] == 0: print(f"W: S{i}, zero nodes. Skip."); skipped_subjects += 1; continue
        #     norm_type = 'l1'
        #     subj_sc_feature = normalize(subj_sc_feature, norm=norm_type, axis=1)
        #     subj_fc_feature = normalize(subj_fc_feature, norm=norm_type, axis=1)
        #     try:
        #         data_sc = get_single_modality_data(subj_sc_adj, subj_sc_feature)
        #         data_fc = get_single_modality_data(subj_fc_adj, subj_fc_feature)
        #     except Exception as e: print(f"E: S{i} create Data: {e}. Skip."); skipped_subjects += 1; continue
        #     data_sc_list.append(data_sc); data_fc_list.append(data_fc); y_list.append(label)
        # print(f"Created initial data lists for {len(data_sc_list)} subjects (skipped {skipped_subjects}).")
        # if not data_sc_list: raise ValueError("No valid subject data found.")

        fc_path = os.path.join(self.raw_dir, self.fc_pkl_filename)
        sc_path = os.path.join(self.raw_dir, self.sc_pkl_filename)
        label_path = os.path.join(self.raw_dir, self.label_pkl_filename)

        print(f"Loading data from: {sc_path}, {fc_path}, {label_path}")

        with open(fc_path, 'rb') as f_fc, open(sc_path, 'rb') as f_sc:
                fc_data = pickle.load(f_fc)
                sc_data = pickle.load(f_sc)

        sc_subjects = set(sc_data.keys())
        fc_subjects = set(fc_data.keys())
        common_subjects = list(sc_subjects.intersection(fc_subjects))
        n_subjects = len(common_subjects)
        print(f"Found {n_subjects} common subjects.")

        # Load labels
        subject_labels = {}
        with open(label_path, 'r', encoding='utf-8') as f_csv:
                csv_reader = csv.DictReader(f_csv)
                for row in csv_reader:
                        subject_key = row['subjectkey']
                        labels = ['Anx', 'OCD', 'ADHD', 'ODD', 'Cond']
                        subject_labels[subject_key] = int(any(row[label] == '1' for label in labels))

        # Create data lists directly instead of processing all_sc_matrices_obj and all_fc_matrices_obj
        data_sc_list = []
        data_fc_list = []
        y_list = []
        skipped_subjects = 0

        for subject in tqdm(common_subjects, desc="Processing subjects"):
                # Skip subjects without labels
                if subject not in subject_labels:
                        print(f"Warning: {subject} has no label, skipping")
                        skipped_subjects += 1
                        continue

                label = torch.LongTensor([subject_labels[subject]])

                # Get SC data for this subject
                try:
                        sc_matrix = sc_data[subject].values if hasattr(sc_data[subject], 'values') else sc_data[subject]
                        if not isinstance(sc_matrix, np.ndarray) or sc_matrix.size == 0: 
                                print(f"W: Subject {subject}, skip SC due to invalid matrix"); 
                                skipped_subjects += 1
                                continue

                        # Ensure it's a 2D matrix
                        if sc_matrix.ndim != 2 or sc_matrix.shape[0] != sc_matrix.shape[1]:
                                print(f"W: Subject {subject}, SC matrix shape issue: {sc_matrix.shape}"); 
                                skipped_subjects += 1
                                continue

                        # Get FC data for this subject
                        fc_matrix = fc_data[subject].values if hasattr(fc_data[subject], 'values') else fc_data[subject]
                        if not isinstance(fc_matrix, np.ndarray) or fc_matrix.size == 0: 
                                print(f"W: Subject {subject}, skip FC due to invalid matrix"); 
                                skipped_subjects += 1
                                continue

                        # Ensure it's a 2D matrix with matching shape
                        if fc_matrix.ndim != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
                                print(f"W: Subject {subject}, FC matrix shape issue: {fc_matrix.shape}"); 
                                skipped_subjects += 1
                                continue

                        # Ensure SC and FC matrices have the same shape
                        if sc_matrix.shape != fc_matrix.shape:
                                print(f"W: Subject {subject}, SC shape {sc_matrix.shape} != FC shape {fc_matrix.shape}"); 
                                skipped_subjects += 1
                                continue

                        # Process matrices
                        sc_feature = sc_matrix.copy()
                        fc_feature = fc_matrix.copy()

                        # Normalize features
                        norm_type = 'l1'
                        sc_feature = normalize(sc_feature, norm=norm_type, axis=1)
                        fc_feature = normalize(fc_feature, norm=norm_type, axis=1)

                        # Create PyG data objects
                        data_sc = get_single_modality_data(sc_matrix, sc_feature)
                        data_fc = get_single_modality_data(fc_matrix, fc_feature)

                        # Add to lists
                        data_sc_list.append(data_sc)
                        data_fc_list.append(data_fc)
                        y_list.append(label)

                except Exception as e: 
                        print(f"Error processing subject {subject}: {e}")
                        skipped_subjects += 1
                        continue

        print(f"Created initial data lists for {len(data_sc_list)} subjects (skipped {skipped_subjects}).")
        if not data_sc_list: 
                raise ValueError("No valid subject data found.")

        


    


        # --- Apply Pre-transform Separately ---
        if self.pre_transform is not None:
            print("Applying pre_transform to SC data...")
            transformed_sc_list = [self.pre_transform(data) for data in tqdm(data_sc_list, desc="Pre-transforming SC")]
            print("Applying pre_transform to FC data...")
            transformed_fc_list = [self.pre_transform(data) for data in tqdm(data_fc_list, desc="Pre-transforming FC")]
            # Filter out None results if transform failed for some graphs
            valid_indices = [i for i, (sc, fc) in enumerate(zip(transformed_sc_list, transformed_fc_list)) if sc is not None and fc is not None]
            if len(valid_indices) < len(transformed_sc_list):
                 print(f"Warning: Pre-transform failed for {len(transformed_sc_list) - len(valid_indices)} subjects. Filtering them out.")
                 transformed_sc_list = [transformed_sc_list[i] for i in valid_indices]
                 transformed_fc_list = [transformed_fc_list[i] for i in valid_indices]
                 y_list = [y_list[i] for i in valid_indices] # Filter labels accordingly
        else:
             print("ERROR: pre_transform is None. HHopSubgraphs is required by the model.")
             transformed_sc_list = data_sc_list
             transformed_fc_list = data_fc_list
             # raise ValueError("Pre-transform (HHopSubgraphs) is required but was not provided.")

        if not transformed_sc_list: raise ValueError("No valid data remaining after pre_transform.")


        # --- Manual Merge and Rename Attributes ---
        final_data_list = []
        print("Merging transformed data and renaming attributes...")
        skipped_merge = 0
        for i, (sc_transformed, fc_transformed) in enumerate(tqdm(zip(transformed_sc_list, transformed_fc_list), total=len(transformed_sc_list), desc="Merging Samples")):

            # --- Create the final Data object for this subject ---
            # merged_data = Data() # OLD
            merged_data = SubgraphData() # NEW: Use the custom class

            # --- Add attributes (Keep the renaming logic as before) ---
            # Check and Rename attributes from SC transform
            if hasattr(sc_transformed, 'original_x'): merged_data.sc_original_x = sc_transformed.original_x
            else: print(f"W: S{i} missing sc original_x"); skipped_merge+=1; continue
            if hasattr(sc_transformed, 'original_edge_index'): merged_data.sc_original_edge_index = sc_transformed.original_edge_index
            else: print(f"W: S{i} missing sc original_edge_index"); skipped_merge+=1; continue
            if hasattr(sc_transformed, 'node_index'): merged_data.sc_node_index = sc_transformed.node_index
            else: print(f"W: S{i} missing sc node_index"); skipped_merge+=1; continue
            if hasattr(sc_transformed, 'edge_index'): merged_data.sc_edge_index = sc_transformed.edge_index # Subgraph edges
            else: print(f"W: S{i} missing sc edge_index"); skipped_merge+=1; continue
            if hasattr(sc_transformed, 'node_to_subgraph'): merged_data.sc_node_to_subgraph = sc_transformed.node_to_subgraph
            else: print(f"W: S{i} missing sc node_to_subgraph"); skipped_merge+=1; continue
            if hasattr(sc_transformed, 'subgraph_to_graph'): merged_data.sc_subgraph_to_graph = sc_transformed.subgraph_to_graph
            else: print(f"W: S{i} missing sc subgraph_to_graph"); skipped_merge+=1; continue
            # Optional attributes
            if hasattr(sc_transformed, 'z'): merged_data.sc_z = sc_transformed.z
            if hasattr(sc_transformed, 'lpe'): merged_data.sc_lpe = sc_transformed.lpe
            if hasattr(sc_transformed, 'x'): merged_data.sc_x = sc_transformed.x # Needed for __inc__
            # else: print(f"DEBUG: S{i} missing sc_transformed.x") # Add temporary debug print

            # Check and Rename attributes from FC transform
            if hasattr(fc_transformed, 'original_x'): merged_data.fc_original_x = fc_transformed.original_x
            else: print(f"W: S{i} missing fc original_x"); skipped_merge+=1; continue
            if hasattr(fc_transformed, 'original_edge_index'): merged_data.fc_original_edge_index = fc_transformed.original_edge_index
            else: print(f"W: S{i} missing fc original_edge_index"); skipped_merge+=1; continue
            if hasattr(fc_transformed, 'node_index'): merged_data.fc_node_index = fc_transformed.node_index
            else: print(f"W: S{i} missing fc node_index"); skipped_merge+=1; continue
            if hasattr(fc_transformed, 'edge_index'): merged_data.fc_edge_index = fc_transformed.edge_index # Subgraph edges
            else: print(f"W: S{i} missing fc edge_index"); skipped_merge+=1; continue
            if hasattr(fc_transformed, 'node_to_subgraph'): merged_data.fc_node_to_subgraph = fc_transformed.node_to_subgraph
            else: print(f"W: S{i} missing fc node_to_subgraph"); skipped_merge+=1; continue
            if hasattr(fc_transformed, 'subgraph_to_graph'): merged_data.fc_subgraph_to_graph = fc_transformed.subgraph_to_graph
            else: print(f"W: S{i} missing fc subgraph_to_graph"); skipped_merge+=1; continue
            # Optional attributes
            if hasattr(fc_transformed, 'z'): merged_data.fc_z = fc_transformed.z
            if hasattr(fc_transformed, 'lpe'): merged_data.fc_lpe = fc_transformed.lpe
            if hasattr(fc_transformed, 'x'): merged_data.fc_x = fc_transformed.x # Needed for __inc__
            # else: print(f"DEBUG: S{i} missing fc_transformed.x") # Add temporary debug print


            # Add label
            merged_data.y = y_list[i]

            # Add num_nodes (use the original node count) - KEEP THIS
            if hasattr(merged_data, 'sc_original_x') and merged_data.sc_original_x is not None and merged_data.sc_original_x.shape[0] > 0:
                 merged_data.num_nodes = merged_data.sc_original_x.shape[0]
            elif hasattr(merged_data, 'fc_original_x') and merged_data.fc_original_x is not None and merged_data.fc_original_x.shape[0] > 0:
                 merged_data.num_nodes = merged_data.fc_original_x.shape[0] # Fallback if SC original_x missing
            else:
                 print(f"\nWarning: Subject index {i} cannot determine original num_nodes after merge (original_x missing or empty). Skipping.")
                 skipped_merge += 1
                 continue # Skip if we can't determine num_nodes reliably


            # --- DEBUGGING PRINT ---
            print(f"\nDEBUG: Subject {i}")
            print(f"  sc_transformed keys: {sc_transformed.keys if sc_transformed is not None else 'None'}")
            print(f"  fc_transformed keys: {fc_transformed.keys if fc_transformed is not None else 'None'}")
            print(f"  merged_data keys before check: {merged_data.keys}")
            # --- END DEBUGGING PRINT ---


            # --- IMPORTANT: Ensure attributes used in __inc__ exist before adding to list ---
            required_attrs = ['sc_x', 'fc_x', 'sc_subgraph_to_graph', 'fc_subgraph_to_graph', 'sc_original_x', 'fc_original_x']
            missing_attrs = [attr for attr in required_attrs if not hasattr(merged_data, attr) or getattr(merged_data, attr) is None]

            if missing_attrs:
                print(f"\nWarning: Subject index {i} missing critical attribute(s) needed for batching increment: {missing_attrs}. Skipping.")
                skipped_merge += 1
                continue

            final_data_list.append(merged_data)

        if not final_data_list:
            print("FATAL ERROR: No subjects remaining after processing and merging. Cannot save empty dataset.")
            raise RuntimeError("No data to save after processing.")

        print(f"Final merged data list created with {len(final_data_list)} samples (skipped {skipped_merge} during merge).")

        # --- Collate and Save ---
        print("Collating final data list...")
        data, slices = self.collate(final_data_list)
        print(f"Collated data object type: {type(data)}")
        if hasattr(data, 'batch'):
            print(f"  'batch' attribute created successfully.")
        else:
            print(f"  WARNING: 'batch' attribute MISSING after collation.")
        print(f"Collated data keys: {data.keys}")

        print(f"Saving processed data to {self.processed_paths[0]}...")
        torch.save((data, slices), self.processed_paths[0])
        print("--- Dataset Processing Complete ---")