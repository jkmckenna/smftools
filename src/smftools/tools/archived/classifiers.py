## Train CNN, RNN, Random Forest models on double barcoded, low contamination datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Device detection
device = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cpu')
)

# ------------------------- Utilities -------------------------
def random_fill_nans(X):
    nan_mask = np.isnan(X)
    X[nan_mask] = np.random.rand(*X[nan_mask].shape)
    return X

# ------------------------- Model Definitions -------------------------
class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        dummy_input = torch.zeros(1, 1, input_size)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _forward_conv(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super().__init__()
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        # Define fully connected output layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))
    
class AttentionRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)  # Simple attention scores
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # shape: (batch, 1, seq_len)
        lstm_out, _ = self.lstm(x)  # shape: (batch, 1, hidden_dim)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, 1, 1)
        context = (attn_weights * lstm_out).sum(dim=1)  # weighted sum
        return self.fc(context)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x
    
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_head = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.input_fc(x).unsqueeze(1)  # -> [batch_size, 1, model_dim]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # -> [seq_len=1, batch_size, model_dim]
        encoded = self.transformer(x)
        pooled = encoded.mean(dim=0)  # -> [batch_size, model_dim]
        return self.cls_head(pooled)

def train_model(model, loader, optimizer, criterion, device, ref_name="", model_name="", epochs=20, patience=5):
    model.train()
    best_loss = float('inf')
    trigger_times = 0

    for epoch in range(epochs):
        running_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(loader)
        print(f"{ref_name} {model_name} Epoch {epoch+1} Loss: {average_loss:.4f}")

        if average_loss < best_loss:
            best_loss = average_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping {model_name} for {ref_name} at epoch {epoch+1}")
                break

def evaluate_model(model, X_tensor, y_encoded, device):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
    fpr, tpr, _ = roc_curve(y_encoded, probs)
    precision, recall, _ = precision_recall_curve(y_encoded, probs)
    f1 = f1_score(y_encoded, preds)
    cm = confusion_matrix(y_encoded, preds)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall,precision)
    # positive-class frequency
    pos_freq = np.mean(y_encoded == 1)
    pr_auc_norm = pr_auc / pos_freq
    return {
        'fpr': fpr, 'tpr': tpr,
        'precision': precision, 'recall': recall,
        'f1': f1, 'auc': roc_auc, 'pr_auc': pr_auc,
        'confusion_matrix': cm, 'pos_freq': pos_freq, 'pr_auc_norm': pr_auc_norm
    }, preds, probs

def train_rf(X_tensor, y_tensor, train_indices, test_indices, n_estimators=500):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    model.fit(X_tensor[train_indices].numpy(), y_tensor[train_indices].numpy())
    probs = model.predict_proba(X_tensor[test_indices].cpu().numpy())[:, 1]
    preds = model.predict(X_tensor[test_indices].cpu().numpy())
    return model, preds, probs

# ------------------------- Main Training Loop -------------------------
def run_training_loop(adata, site_config, layer_name=None,
                       mlp=False, cnn=False, rnn=False, arnn=False, transformer=False, rf=False, nb=False, rr_bayes=False,
                       max_epochs=10, max_patience=5, n_estimators=500, training_split=0.5):
    device = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cpu'))
    metrics, models, positions, tensors = {}, {}, {}, {}
    adata.obs["used_for_training"] = False  # Initialize column to False

    for ref in adata.obs['Reference_strand'].cat.categories:
        ref_subset = adata[adata.obs['Reference_strand'] == ref].copy()
        if ref_subset.shape[0] == 0:
            continue

        # Get matrix and coordinates
        if layer_name:
            matrix = ref_subset.layers[layer_name].copy()
            coords = ref_subset.var_names
            suffix = layer_name
        else:
            site_mask = np.zeros(ref_subset.shape[1], dtype=bool)
            if ref in site_config:
                for site in site_config[ref]:
                    site_mask |= ref_subset.var[f'{ref}_{site}']
                suffix = "_".join(site_config[ref])
            else:
                site_mask = np.ones(ref_subset.shape[1], dtype=bool)
                suffix = "full"
            site_subset = ref_subset[:, site_mask].copy()
            matrix = site_subset.X
            coords = site_subset.var_names

        positions.setdefault(ref, {})[suffix] = coords

        # Fill and encode
        X = random_fill_nans(matrix)
        y = ref_subset.obs["activity_status"]
        y_encoded = y.map({'Active': 1, 'Silent': 0}) 
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded.values, dtype=torch.long)
        tensors.setdefault(ref, {})[suffix] = X_tensor

        # Setup datasets
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(training_split * len(dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        train_indices = adata.obs[adata.obs['Reference_strand'] == ref].index[train_dataset.indices]
        adata.obs.loc[train_indices, "used_for_training"] = True
        
        test_X = X_tensor[test_dataset.indices]
        test_y = y_encoded.iloc[test_dataset.indices] if hasattr(y_encoded, 'iloc') else y_encoded[test_dataset.indices]

        class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        metrics[ref], models[ref] = {}, {}
        
        # MLP
        if mlp:
            mlp_model = MLPClassifier(X.shape[1], len(np.unique(y_encoded))).to(device)
            optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)
            criterion_mlp = nn.CrossEntropyLoss(weight=class_weights)
            train_model(mlp_model, train_loader, optimizer_mlp, criterion_mlp, device, ref, 'MLP', epochs=max_epochs, patience=max_patience)
            mlp_metrics, mlp_preds, mlp_probs = evaluate_model(mlp_model, test_X, test_y, device)
            metrics[ref][f'mlp_{suffix}'] = mlp_metrics
            models[ref][f'mlp_{suffix}'] = mlp_model

        # CNN
        if cnn:
            cnn_model = CNNClassifier(X.shape[1], len(np.unique(y_encoded))).to(device)
            optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)
            criterion_cnn = nn.CrossEntropyLoss(weight=class_weights)
            train_model(cnn_model, train_loader, optimizer_cnn, criterion_cnn, device, ref, 'CNN', epochs=max_epochs, patience=max_patience)
            cnn_metrics, cnn_preds, cnn_probs = evaluate_model(cnn_model, test_X, test_y, device)
            metrics[ref][f'cnn_{suffix}'] = cnn_metrics
            models[ref][f'cnn_{suffix}'] = cnn_model

        # RNN
        if rnn:
            rnn_model = RNNClassifier(X.shape[1], 64, len(np.unique(y_encoded))).to(device)
            optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=0.001)
            criterion_rnn = nn.CrossEntropyLoss(weight=class_weights)
            train_model(rnn_model, train_loader, optimizer_rnn, criterion_rnn, device, ref, 'RNN', epochs=max_epochs, patience=max_patience)
            rnn_metrics, rnn_preds, rnn_probs = evaluate_model(rnn_model, test_X, test_y, device)
            metrics[ref][f'rnn_{suffix}'] = rnn_metrics
            models[ref][f'rnn_{suffix}'] = rnn_model
        
        # Attention RNN
        if arnn:
            arnn_model = AttentionRNNClassifier(X.shape[1], 64, len(np.unique(y_encoded))).to(device)
            optimizer_arnn = optim.Adam(arnn_model.parameters(), lr=0.001)
            criterion_arnn = nn.CrossEntropyLoss(weight=class_weights)
            train_model(arnn_model, train_loader, optimizer_arnn, criterion_arnn, device, ref, 'aRNN', epochs=max_epochs, patience=max_patience)
            arnn_metrics, arnn_preds, arnn_probs = evaluate_model(arnn_model, test_X, test_y, device)
            metrics[ref][f'arnn_{suffix}'] = arnn_metrics
            models[ref][f'arnn_{suffix}'] = arnn_model
                                            
        # Transformer
        if transformer:
            t_model = TransformerClassifier(X.shape[1], 64, len(np.unique(y_encoded))).to(device)
            optimizer_t = optim.Adam(t_model.parameters(), lr=0.001)
            criterion_t = nn.CrossEntropyLoss(weight=class_weights)
            train_model(t_model, train_loader, optimizer_t, criterion_t, device, ref, 'Transformer', epochs=max_epochs, patience=max_patience)
            t_metrics, t_preds, t_probs = evaluate_model(t_model, test_X, test_y, device)
            metrics[ref][f'transformer_{suffix}'] = t_metrics
            models[ref][f'transformer_{suffix}'] = t_model

        # RF
        if rf:
            rf_model, rf_preds, rf_probs = train_rf(X_tensor, y_tensor, train_dataset.indices, test_dataset.indices, n_estimators)
            fpr, tpr, _ = roc_curve(test_y, rf_probs)
            precision, recall, _ = precision_recall_curve(test_y, rf_probs)
            f1 = f1_score(test_y, rf_preds)
            cm = confusion_matrix(test_y, rf_preds)
            roc_auc = auc(fpr, tpr)
            pr_auc = auc(recall, precision)
            metrics[ref][f'rf_{suffix}'] = {
                'fpr': fpr, 'tpr': tpr,
                'precision': precision, 'recall': recall,
                'f1': f1, 'auc': roc_auc, 'confusion_matrix': cm, 'pr_auc': pr_auc
            }
            models[ref][f'rf_{suffix}'] = rf_model

        # Naive Bayes
        if nb:
            nb_model = GaussianNB()
            nb_model.fit(X_tensor[train_dataset.indices].numpy(), y_tensor[train_dataset.indices].numpy())
            nb_probs = nb_model.predict_proba(test_X.numpy())[:, 1]
            nb_preds = nb_model.predict(test_X.numpy())
            fpr_nb, tpr_nb, _ = roc_curve(test_y, nb_probs)
            prec_nb, rec_nb, _ = precision_recall_curve(test_y, nb_probs)
            f1_nb = f1_score(test_y, nb_preds)
            cm_nb = confusion_matrix(test_y, nb_preds)
            auc_nb = auc(fpr_nb, tpr_nb)
            pr_auc_nb = auc(rec_nb, prec_nb)
            metrics[ref][f'nb_{suffix}'] = {
                'fpr': fpr_nb, 'tpr': tpr_nb,
                'precision': prec_nb, 'recall': rec_nb,
                'f1': f1_nb, 'auc': auc_nb, 'confusion_matrix': cm_nb, 'pr_auc': pr_auc_nb
            }
            models[ref][f'nb_{suffix}'] = nb_model

        # Relative-Risk Bayesian
        if rr_bayes:
            # compute relative risks from training
            X_train = X_tensor[train_dataset.indices].numpy()
            y_train_arr = y_tensor[train_dataset.indices].numpy()
            n_a = (y_train_arr==1).sum(); n_s = (y_train_arr==0).sum()
            p_a = (X_train[y_train_arr==1]==1).sum(axis=0)/(n_a+1e-6)
            p_s = (X_train[y_train_arr==0]==1).sum(axis=0)/(n_s+1e-6)
            rr = (p_a+1e-6)/(p_s+1e-6)
            log_rr = np.log(rr)
            # score test
            scores = test_X.numpy().dot(log_rr)
            probs = 1/(1+np.exp(-scores))
            preds = (probs>=0.5).astype(int)
            fpr, tpr, _ = roc_curve(test_y, probs)
            pr, rc, _ = precision_recall_curve(test_y, probs)
            roc_auc = auc(fpr, tpr)
            pr_auc = auc(rc, pr)
            pos_freq = np.mean(test_y==1)
            pr_norm = pr_auc/pos_freq if pos_freq>0 else np.nan
            metrics[ref][f'rr_bayes_{suffix}'] = {
                'fpr': fpr, 'tpr': tpr,
                'precision': pr, 'recall': rc,
                'auc': roc_auc, 'pr_auc': pr_auc, 'pr_auc_norm': pr_norm
            }
            # save rr_bayes parameters as a pseudo-model
            models[ref][f'rr_bayes_{suffix}'] = {
                'log_rr': log_rr,
                'p_active': p_a,
                'p_silent': p_s
            }

    return metrics, models, positions, tensors

def sliding_window_train_test(
    adata,
    site_config,
    layer_name,
    window_sizes,
    step_size,
    training_split=0.7,
    mlp=False,
    cnn=False,
    rnn=False,
    arnn=False,
    transformer=False,
    rf=False,
    nb=False,
    rr_bayes=False,
    epochs=10,
    patience=5,
    batch_size=64,
    n_estimators=500,
    balance_rf_class_weights=True,
    positive_amount=None,
    positive_freq=None,
    bins=None
):
    """
    Slide a window along features, train/test selected models at each position,
    and append ROC-AUC and AUPRC values to the window center in adata.var in a single pass.

    Torch models use GPU (CUDA > MPS > CPU).

    bins: dict mapping bin_name -> boolean mask over adata.obs
    """
    # device detection
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu')
    )

    # define bins: default single bin 'all'
    bin_dict = bins if bins is not None else {'all': np.ones(adata.n_obs, dtype=bool)}

    for bin_name, mask_obs in bin_dict.items():
        mask_array = np.asarray(mask_obs)
        if mask_array.shape[0] != adata.n_obs:
            raise ValueError(f"Mask for bin '{bin_name}' length {mask_array.shape[0]} != n_obs {adata.n_obs}")
        adata_bin = adata[mask_array]
        if adata_bin.n_obs == 0:
            continue

        for ref in adata_bin.obs['Reference_strand'].cat.categories:
            subset = adata_bin[adata_bin.obs['Reference_strand'] == ref]
            if subset.n_obs == 0:
                continue

            # build full feature matrix and var names
            if layer_name:
                suffix = layer_name
                X_full = subset.layers[layer_name].copy()
                var_names = subset.var_names
            else:
                mask_vars = np.zeros(subset.n_vars, dtype=bool)
                if ref in site_config:
                    for site in site_config[ref]:
                        mask_vars |= subset.var[f'{ref}_{site}']
                    suffix = "_".join(site_config[ref])
                else:
                    mask_vars[:] = True
                    suffix = "full"
                X_full = subset[:, mask_vars].X.copy()
                var_names = subset.var_names[mask_vars]

            for window_size in window_sizes:
                # prepare global arrays for each model/metric
                n_vars_global = adata.n_vars
                arrays = {}

                for mname in ['mlp','cnn','rnn','arnn','transformer','rf','nb', 'rr_bayes']:
                    if locals()[mname]:  # if model enabled
                        col_roc = f"{bin_name}_{mname}_{suffix}_w{window_size}_roc"
                        col_pr = f"{bin_name}_{mname}_{suffix}_w{window_size}_pr"
                        col_pr_norm = f"{bin_name}_{mname}_{suffix}_w{window_size}_pr_norm"
                        arrays[(mname,'roc')] = np.full(n_vars_global, np.nan)
                        arrays[(mname,'pr')] = np.full(n_vars_global, np.nan)
                        arrays[(mname,'pr_norm')] = np.full(n_vars_global, np.nan)

                # fill missing and labels
                X_full = random_fill_nans(X_full)
                y_full = subset.obs['activity_status'].map({'Active':1,'Silent':0}).values
                n_feats = X_full.shape[1]

                # sliding windows
                for start in range(0, n_feats - window_size + 1, step_size):
                    end = start + window_size
                    X_win = X_full[:, start:end]
                    center_idx = start + window_size // 2
                    center_var = var_names[center_idx]
                    var_idx_global = adata.var_names.get_loc(center_var)

                    # train/test split
                    X_train, X_pool, y_train, y_pool = train_test_split(
                        X_win, y_full, train_size=training_split,
                        stratify=y_full, random_state=42
                    )

                    # Optional test sampling with fallback
                    try:
                        if positive_amount is not None and positive_freq is not None:
                            pos_idx = np.where(y_pool == 1)[0]
                            if positive_amount > len(pos_idx):
                                raise ValueError("positive_amount exceeds available positives")
                            chosen_pos = np.random.choice(pos_idx, positive_amount, replace=False)

                            neg_amount = int(round(positive_amount * (1 - positive_freq) / positive_freq))
                            neg_idx = np.where(y_pool == 0)[0]
                            if neg_amount > len(neg_idx):
                                raise ValueError("negative_amount exceeds available negatives")
                            chosen_neg = np.random.choice(neg_idx, neg_amount, replace=False)

                            sel = np.concatenate([chosen_pos, chosen_neg])
                            X_test, y_test = X_pool[sel], y_pool[sel]
                        else:
                            X_test, y_test = X_pool, y_pool
                    except ValueError as e:
                        warnings.warn(
                            f"Falling back to full pool for window {start}:{end} ({e})"
                        )
                        X_test, y_test = X_pool, y_pool

                    # prepare DataLoader
                    train_ds = TensorDataset(
                        torch.tensor(X_train, dtype=torch.float32).to(device),
                        torch.tensor(y_train, dtype=torch.long).to(device)
                    )
                    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                    class_w = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                    class_w = torch.tensor(class_w, dtype=torch.float32).to(device)

                    # container for this window's metrics
                    results = {}
                    # train and evaluate models
                    if mlp:
                        model = MLPClassifier(window_size,2).to(device)
                        train_model(model, loader, torch.optim.Adam(model.parameters()), nn.CrossEntropyLoss(weight=class_w), device, ref, f"{bin_name}_MLP", epochs, patience)
                        mets,_,_ = evaluate_model(model, torch.tensor(X_test,dtype=torch.float32).to(device), y_test, device)
                        results['mlp'] = mets
                    if cnn:
                        model = CNNClassifier(window_size,2).to(device)
                        train_model(model, loader, torch.optim.Adam(model.parameters()), nn.CrossEntropyLoss(weight=class_w), device, ref, f"{bin_name}_CNN", epochs, patience)
                        mets,_,_ = evaluate_model(model, torch.tensor(X_test,dtype=torch.float32).to(device), y_test, device)
                        results['cnn'] = mets
                    if rnn:
                        model = RNNClassifier(window_size,64,2).to(device)
                        train_model(model, loader, torch.optim.Adam(model.parameters()), nn.CrossEntropyLoss(weight=class_w), device, ref, f"{bin_name}_RNN", epochs, patience)
                        mets,_,_ = evaluate_model(model, torch.tensor(X_test,dtype=torch.float32).to(device), y_test, device)
                        results['rnn'] = mets
                    if arnn:
                        model = AttentionRNNClassifier(window_size,64,2).to(device)
                        train_model(model, loader, torch.optim.Adam(model.parameters()), nn.CrossEntropyLoss(weight=class_w), device, ref, f"{bin_name}_aRNN", epochs, patience)
                        mets,_,_ = evaluate_model(model, torch.tensor(X_test,dtype=torch.float32).to(device), y_test, device)
                        results['arnn'] = mets
                    if transformer:
                        model = TransformerClassifier(window_size,64,2).to(device)
                        train_model(model, loader, torch.optim.Adam(model.parameters()), nn.CrossEntropyLoss(weight=class_w), device, ref, f"{bin_name}_Transformer", epochs, patience)
                        mets,_,_ = evaluate_model(model, torch.tensor(X_test,dtype=torch.float32).to(device), y_test, device)
                        results['transformer'] = mets
                    if rf:
                        if balance_rf_class_weights:
                            rf_mod = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
                        else:
                            rf_mod = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                        rf_mod.fit(X_train,y_train)
                        probs = rf_mod.predict_proba(X_test)[:,1]
                        fpr,tpr,_ = roc_curve(y_test, probs)
                        pr,rc,_ = precision_recall_curve(y_test, probs)
                        pr_auc = auc(rc,pr)
                        # positive-class frequency
                        pos_freq = np.mean(y_test == 1)
                        pr_auc_norm = pr_auc / pos_freq
                        results['rf'] = {'auc':auc(fpr,tpr),'pr_auc':pr_auc, 'pr_auc_norm': pr_auc_norm}
                    if nb:
                        nb_mod = GaussianNB()
                        nb_mod.fit(X_train,y_train)
                        probs = nb_mod.predict_proba(X_test)[:,1]
                        fpr,tpr,_ = roc_curve(y_test, probs)
                        pr,rc,_ = precision_recall_curve(y_test, probs)
                        pr_auc = auc(rc,pr)
                        # positive-class frequency
                        pos_freq = np.mean(y_test == 1)
                        pr_auc_norm = pr_auc / pos_freq
                        results['nb'] = {'auc':auc(fpr,tpr),'pr_auc':pr_auc, 'pr_auc_norm': pr_auc_norm}
                    if rr_bayes:
                        # Relative-risk Bayesian classifier
                        n_active = np.sum(y_train == 1)
                        n_silent = np.sum(y_train == 0)
                        # compute feature-wise rates
                        p_a = (X_train[y_train == 1] == 1).sum(axis=0) / (n_active + 1e-6)
                        p_s = (X_train[y_train == 0] == 1).sum(axis=0) / (n_silent + 1e-6)
                        rr = (p_a + 1e-6) / (p_s + 1e-6)
                        log_rr = np.log(rr)
                        # score samples
                        scores = X_test.dot(log_rr)
                        probs = 1 / (1 + np.exp(-scores))
                        preds = (probs >= 0.5).astype(int)
                        # metrics
                        fpr, tpr, _ = roc_curve(y_test, probs)
                        pr, rc, _ = precision_recall_curve(y_test, probs)
                        roc_auc = auc(fpr, tpr)
                        pr_auc = auc(rc, pr)
                        pos_freq = np.mean(y_test == 1)
                        pr_norm = pr_auc / pos_freq if pos_freq > 0 else np.nan
                        results['rr_bayes'] = {'auc': roc_auc, 'pr_auc': pr_auc, 'pr_auc_norm': pr_norm}

                    # assign metrics into arrays
                    for mname, mets in results.items():
                        arrays[(mname,'roc')][var_idx_global] = mets['auc']
                        arrays[(mname,'pr')][var_idx_global] = mets['pr_auc']
                        arrays[(mname,'pr_norm')][var_idx_global] = mets['pr_auc_norm']

                # after all windows, write arrays to adata.var
                for (mname,metric), arr in arrays.items():
                    suffix_col = metric
                    col = f"{bin_name}_{mname}_{suffix}_w{window_size}_{suffix_col}"
                    adata.var[col] = arr

    print("✅ Sliding-window training/testing complete. Metrics stored at window centers in adata.var.")

# ------------------------- Apply models to input adata -------------------------
def run_inference(adata, models, site_config, layer_name=None, model_names=["cnn", "mlp", "rf", "nb", "rr_bayes"]):
    """
    Perform inference on the full AnnData object using pre-trained models.
    
    Parameters:
        adata (AnnData): The full AnnData object.
        models (dict): Dictionary of trained models keyed by reference strand.
        site_config (dict): Configuration dictionary for subsetting features by site.
        layer_name (str, optional): Name of the layer to use if applicable.
        model_names (list, optional): List of model names to run inference on.
                                      Defaults to ["cnn", "mlp", "rf", "nb"].
                                      
    Returns:
        None. The function updates adata.obs with predicted class labels, predicted probabilities,
        and active probabilities for each model.
    """
    import numpy as np
    import pandas as pd

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    
    # Loop over each reference key in the models dictionary
    for ref in models.keys():
        # Subset the full AnnData by the reference strand
        full_subset = adata[adata.obs['Reference_strand'] == ref]
        if full_subset.shape[0] == 0:
            continue

        # Reconstruct the same layer or site mask used during training
        if layer_name:
            suffix = layer_name
            full_matrix = full_subset.layers[layer_name].copy()
        else:
            site_mask = np.zeros(full_subset.shape[1], dtype=bool)
            if ref in site_config:
                for site in site_config[ref]:
                    site_mask |= full_subset.var[f'{ref}_{site}']
                suffix = "_".join(site_config[ref])
            else:
                site_mask = np.ones(full_subset.shape[1], dtype=bool)
                suffix = "full"
            full_matrix = full_subset[:, site_mask].X.copy()
        
        # Fill any NaNs in the feature matrix
        full_matrix = random_fill_nans(full_matrix)
        
        # Convert to a torch tensor; for torch models we use the specified device
        full_tensor = torch.tensor(full_matrix, dtype=torch.float32)
        full_tensor_device = full_tensor.to(device)
        
        for model_name in model_names:
            model_key = f"{model_name}_{suffix}"
            pred_col = f"{model_name}_activity_prediction_{suffix}"
            pred_prob_col = f"{model_name}_prediction_prob_{suffix}"
            active_prob_col = f"{model_name}_active_prob_{suffix}"

            if model_key in models[ref]:
                model = models[ref][model_key]
                if model_name in ["rf", "nb"]:
                    # For scikit-learn based models, work on CPU using NumPy arrays
                    X_input = full_tensor.cpu().numpy()
                    preds = model.predict(X_input)
                    probs = model.predict_proba(X_input)
                elif model_name=='rr_bayes':
                    # model is dict of params
                    log_rr = model['log_rr']
                    scores = full_tensor.cpu().numpy().dot(log_rr)
                    probs1 = 1/(1+np.exp(-scores))
                    preds = (probs1>=0.5).astype(int)
                    probs = np.vstack([1-probs1, probs1]).T
                else:
                    # For torch models, perform inference on the specified device
                    model.eval()
                    with torch.no_grad():
                        logits = model(full_tensor_device)
                        preds = logits.argmax(dim=1).cpu().numpy()
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                pred_probs = probs[np.arange(len(preds)), preds]
                active_probs = probs[:, 1]  # class 1 is assumed to be "Active"
                
                # Store predictions in the AnnData object
                adata.obs.loc[full_subset.obs.index, pred_col] = preds
                adata.obs.loc[full_subset.obs.index, pred_prob_col] = pred_probs
                adata.obs.loc[full_subset.obs.index, active_prob_col] = active_probs

        # Ensure that the prediction columns are of categorical type.
        # Replace non-finite values with 0 before converting.
        for model_name in model_names:
            pred_col = f"{model_name}_activity_prediction_{suffix}"
            if pred_col in adata.obs.columns:
                # Convert to numeric (non-finite become NaN), fill NaNs with 0, then cast
                adata.obs[pred_col] = pd.to_numeric(adata.obs[pred_col], errors='coerce').fillna(0).astype(int).astype("category")
    
    print("✅ Inference complete: stored predicted class, predicted probability, and active probability for each model.")

# ------------------------- Evaluate model activity predictions within categorical subgroups -------------------------

def evaluate_model_by_subgroups(
    adata,
    model_prefix="mlp",
    suffix="GpC_site_CpG_site",
    groupby_cols=["Sample_Names_Full", "Enhancer_Open", "Promoter_Open"],
    label_col="activity_status",
    min_samples=10,
    exclude_training_data=True):
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    results = []

    if exclude_training_data:
        test_subset = adata[adata.obs['used_for_training'] == False]
    else:
        test_subset = adata

    df = test_subset.obs.copy()
    df[label_col] = df[label_col].astype('category').cat.codes

    pred_col = f"{model_prefix}_activity_prediction_{suffix}"
    prob_col = f"{model_prefix}_activity_prob_{suffix}"

    if pred_col not in df or prob_col not in df:
        raise ValueError(f"Missing prediction/probability columns for {model_prefix}")

    for group_vals, group_df in df.groupby(groupby_cols):
        if len(group_df) < min_samples:
            continue  # skip small groups

        y_true = group_df[label_col].values
        y_pred = group_df[pred_col].astype(int).values
        y_prob = group_df[prob_col].astype(float).values

        if len(set(y_true)) < 2:
            auc = float('nan')  # undefined if only one class present
        else:
            auc = roc_auc_score(y_true, y_prob)

        results.append({
            **dict(zip(groupby_cols, group_vals)),
            "model": model_prefix,
            "n_samples": len(group_df),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": auc,
        })

    return pd.DataFrame(results)

def evaluate_models_by_subgroup(adata, model_prefixes, groupby_cols, label_col, exclude_training_data=True):
    import pandas as pd
    all_metrics = []
    for model_prefix in model_prefixes:
        try:
            df = evaluate_model_by_subgroups(adata, model_prefix=model_prefix, suffix="GpC_site_CpG_site", groupby_cols=groupby_cols, label_col=label_col, exclude_training_data=exclude_training_data)
            all_metrics.append(df)
        except Exception as e:
            print(f"Skipping {model_prefix} due to error: {e}")

    final_df = pd.concat(all_metrics, ignore_index=True)
    return final_df

def prepare_melted_model_data(adata, outkey='melted_model_df', groupby=['Enhancer_Open', 'Promoter_Open'], label_col='activity_status', model_names = ['cnn', 'mlp', 'rf'], suffix='GpC_site_CpG_site', omit_training=True):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    cols = groupby.append(label_col)
    if omit_training:
        subset = adata[adata.obs['used_for_training'] == False]
    else:
        subset = adata
    df = subset.obs[cols].copy()
    df[label_col] = df[label_col].astype('category').cat.codes

    for model in model_names:
        col = f"{model}_active_prob_{suffix}"
        if col in subset.obs.columns:
            df[f"{model}_prob"] = subset.obs[col].astype(float)

    # Melt into long format
    melted = df.melt(
        id_vars=cols,
        value_vars=[f"{m}_prob" for m in model_names if f"{m}_active_prob_{suffix}" in subset.obs.columns],
        var_name='model',
        value_name='prob'
    )
    melted['model'] = melted['model'].str.replace('_prob', '', regex=False)
    
    adata.uns[outkey] = melted
