
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def plot_model_performance(metrics, save_path=None):
    import matplotlib.pyplot as plt
    import os
    for ref in metrics.keys():
        plt.figure(figsize=(12, 5))

        # ROC Curve
        plt.subplot(1, 2, 1)
        for model_name, vals in metrics[ref].items():
            model_type = model_name.split('_')[0]
            data_type = model_name.split(f"{model_type}_")[1]
            plt.plot(vals['fpr'], vals['tpr'], label=f"{model_type.upper()} - AUC: {vals['auc']:.4f}")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{data_type} ROC Curve ({ref})')
        plt.legend()

        # PR Curve
        plt.subplot(1, 2, 2)
        for model_name, vals in metrics[ref].items():
            model_type = model_name.split('_')[0]
            data_type = model_name.split(f"{model_type}_")[1]
            plt.plot(vals['recall'], vals['precision'], label=f"{model_type.upper()} - F1: {vals['f1']:.4f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{data_type} Precision-Recall Curve ({ref})')
        plt.legend()

        plt.tight_layout()

        if save_path:
            save_name = f"{ref}"
            os.makedirs(save_path, exist_ok=True)
            safe_name = save_name.replace("=", "").replace("__", "_").replace(",", "_")
            out_file = os.path.join(save_path, f"{safe_name}.png")
            plt.savefig(out_file, dpi=300)
            print(f"📁 Saved: {out_file}")
        plt.show()
        
        # Confusion Matrices
        for model_name, vals in metrics[ref].items():
            print(f"Confusion Matrix for {ref} - {model_name.upper()}:")
            print(vals['confusion_matrix'])
            print()

def plot_feature_importances_or_saliency(models, positions, tensors, site_config, layer_name=None, save_path=None):
    """
    For each reference in the models dictionary, plot:
      - For Random Forest (rf) models: feature importances.
      - For neural network (e.g. mlp, cnn) models: feature saliency computed via input gradients.
    """
    # Select device for NN models
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    
    # Loop over references in models
    for ref, model_dict in models.items():
        # Determine the suffix used for this reference.
        if layer_name:
            suffix = layer_name
        else:
            # If site_config exists for this ref, join the list to form the suffix; else, use "full"
            if ref in site_config:
                suffix = "_".join(site_config[ref])
            else:
                suffix = "full"
                
        # Get genomic positions from positions dictionary.
        if ref not in positions or suffix not in positions[ref]:
            print(f"Positions not found for {ref} with suffix {suffix}. Skipping {ref}.")
            continue
        coords = positions[ref][suffix].astype(int)
        
        # Loop over each model in the sub-dictionary that matches the suffix.
        for model_key, model in model_dict.items():
            # Only consider models with keys ending in the same suffix.
            if not model_key.endswith(suffix):
                continue
            
            # Decide whether we're dealing with a Random Forest or a neural network.
            if model_key.startswith("rf"):
                # Random Forest: use built-in feature importances.
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                else:
                    print(f"Random Forest model {model_key} has no feature_importances_. Skipping.")
                    continue
                plot_title = f"RF Feature Importances for {ref} ({model_key})"
                y_label = "Feature Importance"
            else:
                # Neural network: compute saliency.
                # Check if input data has been provided.
                if tensors is None:
                    print(f"No input data provided for NN saliency for {model_key}. Skipping.")
                    continue
                if ref not in tensors or suffix not in tensors[ref]:
                    print(f"Input data not found for {ref} with suffix {suffix}. Skipping {model_key}.")
                    continue
                input_tensor = tensors[ref][suffix]
                
                # Move model and input to device, and set input to require gradients.
                model.eval()
                input_tensor = input_tensor.to(device)
                input_tensor.requires_grad_()
                
                # Forward pass and compute gradient-based saliency.
                # Here we choose class index 1 (Active) as target and sum over samples.
                with torch.enable_grad():
                    logits = model(input_tensor)
                    score = logits[:, 1].sum()
                    score.backward()
                    # Average gradient magnitude over the batch gives a saliency per feature.
                    saliency = input_tensor.grad.abs().mean(dim=0).cpu().numpy()
                importances = saliency
                plot_title = f"Feature Saliency for {ref} ({model_key})"
                y_label = "Feature Saliency"
            
            # Sort positions and associated importances.
            sorted_idx = np.argsort(coords)
            positions_sorted = coords[sorted_idx]
            importances_sorted = np.array(importances)[sorted_idx]
            
            # Plot the result.
            plt.figure(figsize=(12, 4))
            plt.plot(positions_sorted, importances_sorted, marker='o', linestyle='-', alpha=0.7)
            plt.xlabel("Genomic Position")
            plt.ylabel(y_label)
            plt.title(plot_title)
            plt.grid(True)
            plt.tight_layout()

            if save_path:
                save_name = f"{plot_title}"
                os.makedirs(save_path, exist_ok=True)
                safe_name = save_name.replace("=", "").replace("__", "_").replace(",", "_")
                out_file = os.path.join(save_path, f"{safe_name}.png")
                plt.savefig(out_file, dpi=300)
                print(f"📁 Saved: {out_file}")

            plt.show()


def plot_model_curves_from_adata(adata, label_col='activity_status', model_names = ["cnn", "mlp", "rf"], suffix='GpC_site_CpG_site', omit_training=True, save_path=None):
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns    

    if omit_training:
        subset = adata[adata.obs['used_for_training'].astype(bool) == False]
    label = subset.obs[label_col].astype('category').cat.codes.values  # Convert to 0/1

    plt.figure(figsize=(12, 5))

    # ROC curve
    plt.subplot(1, 2, 1)
    for model in model_names:
        prob_col = f"{model}_active_prob_{suffix}"
        if prob_col in subset.obs.columns:
            probs = subset.obs[prob_col].astype(float).values
            fpr, tpr, _ = roc_curve(label, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{model.upper()} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # PR curve
    plt.subplot(1, 2, 2)
    for model in model_names:
        prob_col = f"{model}_active_prob_{suffix}"
        if prob_col in subset.obs.columns:
            probs = subset.obs[prob_col].astype(float).values
            precision, recall, _ = precision_recall_curve(label, probs)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"{model.upper()} (AUC={pr_auc:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()
    if save_path:
        save_name = f"ROC_PR_curves"
        os.makedirs(save_path, exist_ok=True)
        safe_name = save_name.replace("=", "").replace("__", "_").replace(",", "_")
        out_file = os.path.join(save_path, f"{safe_name}.png")
        plt.savefig(out_file, dpi=300)
        print(f"📁 Saved: {out_file}")
    plt.show()
