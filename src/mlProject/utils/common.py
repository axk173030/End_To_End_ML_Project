from ensure import ensure_annotations
from box import ConfigBox
from typing import Any, Dict, List, Optional, Tuple, Union
import os
from box.exceptions import BoxValueError
from mlProject import logger
import json
import joblib
from pathlib import Path


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.
    
    Args:
        path_to_yaml (str): Path to the YAML file.
    
    Returns:
        ConfigBox: Content of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml_file.read()
            logger(f"YAML file read successfully: {path_to_yaml}")
            return ConfigBox.from_yaml(content)
    except FileNotFoundError as e:
        logger.error(f"File not found: {path_to_yaml}")
        raise e
    except BoxValueError as e:
        logger.error(f"Error reading to YAML file: {e}")
        raise e
    
    @ensure_annotations
    def create_directories(path_to_dirs: List, verbose: bool = True) -> None:
        """
        Creates directories if they do not exist.
        
        Args:
            path_to_dirs (List): List of directory paths to create.
            verbose (bool): If True, prints the created directories.
        """
        for path in path_to_dirs:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger(f"Directory created: {path}")
                

    @ensure_annotations
    def save_json(path: Path, data: Dict) -> None:
        """
        Saves a dictionary as a JSON file.
        
        Args:
            path (Path): Path to save the JSON file.
            data (Dict): Dictionary to save.
        """
        with open(path, "w") as f:
            json.dump(data, f ,indent=4)
            logger(f"JSON file saved successfully: {path}")            

    @ensure_annotations
    def load_json(path: Path) -> Dict:
        """
        Loads a JSON file and returns its content as a dictionary.
        
        Args:
            path (Path): Path to the JSON file.
        
        Returns:
            Dict: Content of the JSON file as a dictionary.
        """
        with open(path, "r") as f:
            data = json.load(f)
            logger(f"JSON file loaded successfully: {path}")
            return data
    @ensure_annotations
    def save_model(path: Path, model: Any) -> None:
        """
        Saves a model using joblib.
        
        Args:
            path (Path): Path to save the model.
            model (Any): Model to save.
        """
        joblib.dump(model, path)
        logger(f"Model saved successfully: {path}")
    @ensure_annotations
    def get_size(path: Path) -> Union[int, float]:
        """
        Returns the size of a file or directory.
        
        Args:
            path (Path): Path to the file or directory.
        
        Returns:
            Union[int, float]: Size of the file or directory in bytes.
        """
        if os.path.isfile(path):
            return os.path.getsize(path)
        elif os.path.isdir(path):
            return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
        else:
            raise ValueError(f"Path does not exist: {path}")
    @ensure_annotations
    def load_binary(path: Path) -> Any:
        """
        Loads a binary file using joblib.
        
        Args:
            path (Path): Path to the binary file.
        
        Returns:
            Any: Loaded object from the binary file.
        """
        model = joblib.load(path)
        logger(f"Binary file loaded successfully: {path}")
        return model
    