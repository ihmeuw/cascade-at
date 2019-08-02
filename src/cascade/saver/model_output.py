from pathlib import Path


def write_model_results(model_results, model_version_id, model_type, output_dir):
    """ Writes the model_results dataframe as a csv file to the output dir.
    """
    file_name = Path(output_dir) / f"{model_type.lower()}_{model_version_id}.csv"
    model_results.to_csv(file_name, index=False)
