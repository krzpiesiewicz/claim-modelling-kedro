def get_file_name(file_template: str, dataset: str) -> str:
    """
    Returns the file name based on the template and dataset name.

    Args:
        file_template (str): Template for the file name.
        dataset (str): Name of the dataset (e.g., "train" or "test").

    Returns:
        str: Formatted file name.
    """
    return file_template.format(dataset=dataset)