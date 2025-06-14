from ..data import AnnDataModule
from ..evaluation import PostInferenceModelEvaluator
from .lightning_inference import run_lightning_inference
from .sklearn_inference import run_sklearn_inference

def sliding_window_inference(
    adata, 
    trained_results, 
    tensor_source='X',
    tensor_key=None,
    label_col='activity_status',
    batch_size=64,
    cleanup=False,
    target_eval_freq=None, 
    max_eval_positive=None
):
    """
    Apply trained sliding window models to an AnnData object (Lightning or Sklearn).
    Evaluate model performance and return a df.
    Optionally remove the appended inference columns from AnnData to clean up obs namespace.
    """
    ## Inference using trained models
    for model_name, model_dict in trained_results.items():
        for window_size, window_data in model_dict.items():
            for center_varname, run in window_data.items():
                print(f"\nEvaluating {model_name} window {window_size} around {center_varname}")
                
                # Extract window start from varname
                center_idx = adata.var_names.get_loc(center_varname)
                window_start = center_idx - window_size // 2
                
                # Build datamodule for window
                datamodule = AnnDataModule(
                    adata,
                    tensor_source=tensor_source,
                    tensor_key=tensor_key,
                    label_col=label_col,
                    batch_size=batch_size,
                    window_start=window_start,
                    window_size=window_size,
                    inference_mode=True
                )
                datamodule.setup()

                # Extract model + detect type
                model = run['model']

                # Lightning models
                if hasattr(run, 'trainer') or 'trainer' in run:
                    trainer = run['trainer']
                    run_lightning_inference(
                        adata,
                        model=model,
                        datamodule=datamodule,
                        trainer=trainer,
                        prefix=f"{model_name}_w{window_size}_c{center_varname}"
                    )
                
                # Sklearn models
                else:
                    run_sklearn_inference(
                        adata,
                        model=model,
                        datamodule=datamodule,
                        prefix=f"{model_name}_w{window_size}_c{center_varname}"
                    )

    print("Inference complete across all models.")

    ## Post-inference model evaluation
    model_wrappers = {}

    for model_name, model_dict in trained_results.items():
        for window_size, window_data in model_dict.items():
            for center_varname, run in window_data.items():
                # Reconstruct the prefix string you used in inference
                prefix = f"{model_name}_w{window_size}_c{center_varname}"
                # Use full key for uniqueness
                key = prefix
                model_wrappers[key] = run['model']

    # Run evaluator
    evaluator = PostInferenceModelEvaluator(adata, model_wrappers, target_eval_freq=target_eval_freq, max_eval_positive=max_eval_positive)
    evaluator.evaluate_all()

    # Get results
    df = evaluator.to_dataframe()

    df[['model_name', 'window_size', 'center']] = df['model'].str.extract(r'(\w+)_w(\d+)_c(\d+)_activity_status')

    # Cast window_size and center to integers for plotting
    df['window_size'] = df['window_size'].astype(int)
    df['center'] = df['center'].astype(int)

    ## Optional cleanup:
    if cleanup:
        prefixes = [f"{model_name}_w{window_size}_c{center_varname}" 
                    for model_name, model_dict in trained_results.items() 
                    for window_size, window_data in model_dict.items() 
                    for center_varname in window_data.keys()]

        # Remove matching obs columns
        for prefix in prefixes:
            to_remove = [col for col in adata.obs.columns if col.startswith(prefix)]
            adata.obs.drop(columns=to_remove, inplace=True)

            # Remove obsm entries if any
            obsm_key = f"{prefix}_pred_prob_all"
            if obsm_key in adata.obsm:
                del adata.obsm[obsm_key]

        print(f"Cleaned up {len(prefixes)} model prefixes from AnnData.")

    return df