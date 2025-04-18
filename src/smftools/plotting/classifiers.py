
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

def plot_feature_importances_or_saliency(
    models,
    positions,
    tensors,
    site_config,
    adata=None,
    layer_name=None,
    save_path=None,
    shaded_regions=None
):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Select device for NN models
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu')
    )

    for ref, model_dict in models.items():
        if layer_name:
            suffix = layer_name
        else:
            suffix = "_".join(site_config[ref]) if ref in site_config else "full"

        if ref not in positions or suffix not in positions[ref]:
            print(f"Positions not found for {ref} with suffix {suffix}. Skipping {ref}.")
            continue

        coords_index = positions[ref][suffix]
        coords = coords_index.astype(int)

        # Classify positions using adata.var columns
        cpg_sites = set()
        gpc_sites = set()
        other_sites = set()

        if adata is None:
            print("⚠️ AnnData object is required to classify site types. Skipping site type markers.")
        else:
            gpc_col = f"{ref}_GpC_site"
            cpg_col = f"{ref}_CpG_site"
            for idx_str in coords_index:
                try:
                    gpc = adata.var.at[idx_str, gpc_col] if gpc_col in adata.var.columns else False
                    cpg = adata.var.at[idx_str, cpg_col] if cpg_col in adata.var.columns else False
                    coord_int = int(idx_str)
                    if gpc and not cpg:
                        gpc_sites.add(coord_int)
                    elif cpg and not gpc:
                        cpg_sites.add(coord_int)
                    else:
                        other_sites.add(coord_int)
                except KeyError:
                    print(f"⚠️ Index '{idx_str}' not found in adata.var. Skipping.")
                    continue

        for model_key, model in model_dict.items():
            if not model_key.endswith(suffix):
                continue

            if model_key.startswith("rf"):
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                else:
                    print(f"Random Forest model {model_key} has no feature_importances_. Skipping.")
                    continue
                plot_title = f"RF Feature Importances for {ref} ({model_key})"
                y_label = "Feature Importance"
            else:
                if tensors is None or ref not in tensors or suffix not in tensors[ref]:
                    print(f"No input data provided for NN saliency for {model_key}. Skipping.")
                    continue
                input_tensor = tensors[ref][suffix]
                model.eval()
                input_tensor = input_tensor.to(device)
                input_tensor.requires_grad_()

                with torch.enable_grad():
                    logits = model(input_tensor)
                    score = logits[:, 1].sum()
                    score.backward()
                    saliency = input_tensor.grad.abs().mean(dim=0).cpu().numpy()
                importances = saliency
                plot_title = f"Feature Saliency for {ref} ({model_key})"
                y_label = "Feature Saliency"

            sorted_idx = np.argsort(coords)
            positions_sorted = coords[sorted_idx]
            importances_sorted = np.array(importances)[sorted_idx]

            plt.figure(figsize=(12, 4))
            for pos, imp in zip(positions_sorted, importances_sorted):
                if pos in cpg_sites:
                    plt.plot(pos, imp, marker='*', color='black', markersize=10, linestyle='None',
                             label='CpG site' if 'CpG site' not in plt.gca().get_legend_handles_labels()[1] else "")
                elif pos in gpc_sites:
                    plt.plot(pos, imp, marker='o', color='blue', markersize=6, linestyle='None',
                             label='GpC site' if 'GpC site' not in plt.gca().get_legend_handles_labels()[1] else "")
                else:
                    plt.plot(pos, imp, marker='.', color='gray', linestyle='None',
                             label='Other' if 'Other' not in plt.gca().get_legend_handles_labels()[1] else "")

            plt.plot(positions_sorted, importances_sorted, linestyle='-', alpha=0.5, color='black')

            if shaded_regions:
                for (start, end) in shaded_regions:
                    plt.axvspan(start, end, color='gray', alpha=0.3)

            plt.xlabel("Genomic Position")
            plt.ylabel(y_label)
            plt.title(plot_title)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                safe_name = plot_title.replace("=", "").replace("__", "_").replace(",", "_").replace(" ", "_")
                out_file = os.path.join(save_path, f"{safe_name}.png")
                plt.savefig(out_file, dpi=300)
                print(f"📁 Saved: {out_file}")

            plt.show()

def plot_model_curves_from_adata(
    adata, 
    label_col='activity_status', 
    model_names = ["cnn", "mlp", "rf"], 
    suffix='GpC_site_CpG_site',
    omit_training=True, 
    save_path=None, 
    ylim_roc=(0.0, 1.05), 
    ylim_pr=(0.0, 1.05)):

    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns    

    if omit_training:
        subset = adata[adata.obs['used_for_training'].astype(bool) == False]

    label = subset.obs[label_col].map({'Active': 1, 'Silent': 0}).values  

    positive_ratio = np.sum(label.astype(int)) / len(label)

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
    plt.ylim(*ylim_roc)
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
    plt.ylim(*ylim_pr)
    plt.axhline(y=positive_ratio, linestyle='--', color='gray', label='Random Baseline')
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

def plot_model_curves_from_adata_with_frequency_grid(
    adata,
    label_col='activity_status',
    model_names=["cnn", "mlp", "rf"],
    suffix='GpC_site_CpG_site',
    omit_training=True,
    save_path=None,
    ylim_roc=(0.0, 1.05),
    ylim_pr=(0.0, 1.05),
    pos_sample_count=500,
    pos_freq_list=[0.01, 0.05, 0.1],
    show_f1_iso_curves=False,
    f1_levels=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    
    if f1_levels is None:
        f1_levels = np.linspace(0.2, 0.9, 8)
        
    if omit_training:
        subset = adata[adata.obs['used_for_training'].astype(bool) == False]
    else:
        subset = adata

    label = subset.obs[label_col].map({'Active': 1, 'Silent': 0}).values
    subset = subset.copy()
    subset.obs["__label__"] = label

    pos_indices = np.where(label == 1)[0]
    neg_indices = np.where(label == 0)[0]

    n_rows = len(pos_freq_list)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
    fig.suptitle(f'{suffix} Performance metrics')

    for row_idx, pos_freq in enumerate(pos_freq_list):
        desired_total = int(pos_sample_count / pos_freq)
        neg_sample_count = desired_total - pos_sample_count

        if pos_sample_count > len(pos_indices) or neg_sample_count > len(neg_indices):
            print(f"⚠️ Skipping frequency {pos_freq:.3f}: not enough samples.")
            continue

        sampled_pos = np.random.choice(pos_indices, size=pos_sample_count, replace=False)
        sampled_neg = np.random.choice(neg_indices, size=neg_sample_count, replace=False)
        sampled_indices = np.concatenate([sampled_pos, sampled_neg])

        data_sampled = subset[sampled_indices]
        y_true = data_sampled.obs["__label__"].values

        ax_roc = axes[row_idx, 0] if n_rows > 1 else axes[0]
        ax_pr = axes[row_idx, 1] if n_rows > 1 else axes[1]

        # ROC Curve
        for model in model_names:
            prob_col = f"{model}_active_prob_{suffix}"
            if prob_col in data_sampled.obs.columns:
                probs = data_sampled.obs[prob_col].astype(float).values
                fpr, tpr, _ = roc_curve(y_true, probs)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, label=f"{model.upper()} (AUC={roc_auc:.4f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_ylim(*ylim_roc)
        ax_roc.set_title(f"ROC Curve (Pos Freq: {pos_freq:.2%})")
        ax_roc.legend()
        ax_roc.spines['top'].set_visible(False)
        ax_roc.spines['right'].set_visible(False)

        # PR Curve
        for model in model_names:
            prob_col = f"{model}_active_prob_{suffix}"
            if prob_col in data_sampled.obs.columns:
                probs = data_sampled.obs[prob_col].astype(float).values
                precision, recall, _ = precision_recall_curve(y_true, probs)
                pr_auc = auc(recall, precision)
                ax_pr.plot(recall, precision, label=f"{model.upper()} (AUC={pr_auc:.4f})")
        ax_pr.axhline(y=pos_freq, linestyle='--', color='gray', label='Random Baseline')

        if show_f1_iso_curves:
            recall_vals = np.linspace(0.01, 1, 500)
            for f1 in f1_levels:
                precision_vals = (f1 * recall_vals) / (2 * recall_vals - f1)
                precision_vals[precision_vals < 0] = np.nan  # Avoid plotting invalid values
                ax_pr.plot(recall_vals, precision_vals, color='gray', linestyle=':', linewidth=1, alpha=0.6)
                x_val = 0.9
                y_val = (f1 * x_val) / (2 * x_val - f1)
                if 0 < y_val < 1:
                    ax_pr.text(x_val, y_val, f"F1={f1:.1f}", fontsize=8, color='gray')

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_ylim(*ylim_pr)
        ax_pr.set_title(f"PR Curve (Pos Freq: {pos_freq:.2%})")
        ax_pr.legend()
        ax_pr.spines['top'].set_visible(False)
        ax_pr.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, "ROC_PR_grid.png")
        plt.savefig(out_file, dpi=300)
        print(f"📁 Saved: {out_file}")
    plt.show()