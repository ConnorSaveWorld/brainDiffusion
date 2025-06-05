import time
import torch
import torch.nn.functional as F
from model_components.loss import SupConLoss
from eval import eval_FCSC, eval_FCSC_for_tsne


import time
import torch
import torch.nn.functional as F
# from model_components.loss import SupConLoss # Assuming not used for DAC_GNet
from eval import eval_FCSC, eval_FCSC_for_tsne


def train_model(args, model, optimizer, scheduler, train_loader, val_loader, test_loader, i_fold):
    """
    :param train_loader:
    :param model: model
    :type optimizer: Optimizer
    """
    max_val_auc = 0.0  # Changed to monitor validation AUC for model saving
    patience_counter = 0 # Renamed from 'patience' to avoid confusion with args.patience
    best_epoch = 0
    t0 = time.time()

    class_weights = None
    

    # --- Calculate Class Weights (once per fold for the current training data) ---
    # if args.model_name == 'DAC_GNet' or 'GNN_Transformer' in args.model_name:
    if 1+1 ==2:    
        if args.num_classes == 2: # Only calculate for binary classification
            all_labels_list = []
        # train_loader.dataset should be the Subset or sliced Dataset for the current fold
            try:
            # Attempt to get labels efficiently if the dataset object has a 'y' attribute
                if hasattr(train_loader.dataset, 'y') and isinstance(train_loader.dataset.y, torch.Tensor):
                    all_labels_tensor_fold = train_loader.dataset.y
                    if all_labels_tensor_fold.ndim > 1: # Ensure it's a 1D tensor of labels
                        all_labels_tensor_fold = all_labels_tensor_fold.view(-1)
                    print(f"Fold {i_fold}: Using train_loader.dataset.y for class weights. Shape: {all_labels_tensor_fold.shape}")

            # Fallback: Iterate through the dataset to collect labels
                else:
                    print(f"Fold {i_fold}: Iterating train_loader.dataset to collect labels for class weights.")
                    for data_item in train_loader.dataset: # Iterates over individual Data objects in the current fold's training subset
                        if isinstance(data_item.y, torch.Tensor) and data_item.y.numel() == 1:
                            all_labels_list.append(data_item.y.item())
                        elif isinstance(data_item.y, (int, float)):
                            all_labels_list.append(data_item.y)
                        else:
                        # If labels are batched even in train_loader.dataset, this needs adjustment
                        # But typically, train_loader.dataset is a list/iterable of single Data objects
                            all_labels_list.extend(data_item.y.view(-1).tolist()) # Assuming data_item.y might be a small tensor
                    if not all_labels_list:
                        raise ValueError("Could not extract any labels from train_loader.dataset")
                    all_labels_tensor_fold = torch.tensor(all_labels_list, dtype=torch.long)

                num_total_fold = all_labels_tensor_fold.numel()
                if num_total_fold > 0:
                    num_class_0_fold = (all_labels_tensor_fold == 0).sum().item()
                    num_class_1_fold = (all_labels_tensor_fold == 1).sum().item()

                    if num_class_0_fold > 0 and num_class_1_fold > 0:
                    # Standard weighting: n_samples / (n_classes * n_samples_for_class_i)
                        weight_class_0 = num_total_fold / (float(args.num_classes) * num_class_0_fold)
                        weight_class_1 = num_total_fold / (float(args.num_classes) * num_class_1_fold)
                        class_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float)
                        if args.use_cuda:
                            class_weights = class_weights.cuda()
                        print(f"Fold {i_fold}: Using class weights: {class_weights} (Counts: C0={num_class_0_fold}, C1={num_class_1_fold})")
                    else:
                        print(f"Fold {i_fold}: Warning: One class is missing in the training data for this fold (C0={num_class_0_fold}, C1={num_class_1_fold}). Using unweighted loss.")
                else:
                    print(f"Fold {i_fold}: Warning: No labels found in training data for this fold. Using unweighted loss.")

            except Exception as e:
                print(f"Fold {i_fold}: Error calculating class weights: {e}. Using unweighted loss.")
                class_weights = None # Fallback to unweighted
        else:
            print(f"Fold {i_fold}: num_classes is not 2 ({args.num_classes}). Using unweighted loss.")
    # --- End Class Weights Calculation ---

    for epoch in range(args.epochs):
        model.train()
        t_epoch_start = time.time()
        train_loss_sum = 0.0
        train_correct_sum = 0
        num_train_samples_epoch = 0

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if args.use_cuda:
                data = data.cuda()
            
            labels = data.y.view(-1) # Ensure labels are 1D

            # --- Modified model output unpacking ---
            model_return_values = model(data)
            
            # Initialize sc_x and fc_x to None. They will be populated if the model returns enough values.
            sc_x, fc_x = None, None 

            if not isinstance(model_return_values, tuple):
                # Model returned a single tensor (assumed to be output/logits)
                output = model_return_values
            else:
                # Model returned a tuple of values
                num_returned = len(model_return_values)
                if num_returned == 0:
                    raise ValueError("Model returned an empty tuple, which is not expected.")
                
                output = model_return_values[0]  # First element is assumed to be logits

                if num_returned >= 3:
                    # If 3 or more values, assign 2nd to sc_x, 3rd to fc_x
                    # This covers models like DAC_GNet (3), Alternately_Attention_Bottlenecks (4), Attention_Bottlenecks (5)
                    sc_x = model_return_values[1]
                    fc_x = model_return_values[2]
                elif num_returned == 2:
                    # Model returned 2 values (e.g., output, some_feature)
                    # The original except block was `output, _ = model(data)`,
                    # implying sc_x and fc_x were not populated from the second item.
                    # So, sc_x and fc_x remain None.
                    # If the second item is intended for sc_x or fc_x, adjust here.
                    # For example: _some_feature = model_return_values[1]
                    pass 
                # If num_returned == 1 (tuple with one item), output is already set, sc_x/fc_x remain None.
            # --- End of modified model output unpacking ---


            # --- Apply class_weights to the loss function ---
            loss_classifier = F.cross_entropy(output, labels, weight=class_weights)
            
            # Your commented-out contrastive loss terms:
            # These would require sc_x, fc_x (or more specific features like single_modal_cls_sc etc.)
            # to be correctly populated and have the expected meaning/shape.
            # loss_single_modal_con = (sup_con_loss(single_modal_cls_sc, labels.view(-1)) + sup_con_loss(single_modal_cls_fc, labels.view(-1))) / 5
            # loss_multi_modal_con = sup_con_loss(torch.cat((cls_sc, cls_fc), dim=1), labels.view(-1)) / 5
            # loss = loss_classifier + args.lamda_1 * loss_single_modal_con + args.lamda_2 * loss_multi_modal_con
            
            loss = loss_classifier # Using only classifier loss as per current active code
            loss.backward()
            optimizer.step()

            pred = output.data.argmax(dim=1)
            train_loss_sum += loss.item() * output.shape[0] # Accumulate loss scaled by batch size
            train_correct_sum += torch.sum(pred == labels).item()
            num_train_samples_epoch += labels.size(0)
        
        scheduler.step()

        avg_train_loss = train_loss_sum / num_train_samples_epoch if num_train_samples_epoch > 0 else 0
        avg_train_acc = train_correct_sum / num_train_samples_epoch if num_train_samples_epoch > 0 else 0

        val_acc, val_loss, val_sen, val_spe, val_f1, val_auc, _, _ = eval_FCSC(args, model, val_loader)
        test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, _, _ = eval_FCSC(args, model, test_loader)
        
        if epoch % 30 == 0 and test_loader is not None:
            # Assuming eval_FCSC_for_tsne internally calls model(data) and handles its outputs.
            # If it relies on sc_x, fc_x being specific features, ensure they are correctly extracted.
            eval_FCSC_for_tsne(args, model, test_loader, epoch)

        print(f'Epoch: {epoch:04d} train_loss: {avg_train_loss:.6f} train_acc: {avg_train_acc:.6f} '
              f'val_loss: {val_loss:.6f} val_acc: {val_acc:.6f} val_auc: {val_auc:.6f} '
              f'test_loss: {test_loss:.6f} test_acc: {test_acc:.6f} test_auc: {test_auc:.6f} '
              f'time: {time.time() - t_epoch_start:.6f}s')

        if val_auc > max_val_auc: # Monitor test_auc as per original logic
            max_val_auc = val_auc
            print(f"Test AUC improved to {max_val_auc:.6f}. Saving model...")
            if test_loader is not None:
                 eval_FCSC_for_tsne(args, model, test_loader, epoch) # Save t-SNE on improvement
            torch.save(model.state_dict(), f'ckpt/{args.dataset}/{i_fold}_fold_best_model.pth')
            print(f"Model saved at epoch {epoch} for fold {i_fold}")
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} due to no improvement in test AUC for {args.patience} epochs.")
            break

    print(f'Optimization Finished for fold {i_fold}! Total time elapsed: {time.time() - t0:.6f}')
    print(f"Best epoch for fold {i_fold} was {best_epoch} with test AUC: {max_val_auc:.6f}") # Changed from validation AUC to test AUC to match saving criteria

    return best_epoch