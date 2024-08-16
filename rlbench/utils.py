import json
import os
import glob
import pickle
from typing import Optional, Dict, List, Any
import shutil

def create_or_clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.mkdir(folder_path)


def clear_folder(folder_path):
    files = glob.glob(f'{folder_path}/*')
    for f in files:
        os.remove(f)


def copy_to_dict(from_dict: Optional[Dict], to_dict: Dict) -> Dict:
    assert from_dict is not None
    assert to_dict is not None

    for key in from_dict.keys():
        to_dict[key] = from_dict[key]
    return to_dict


def join_paths(*paths):
    paths = list(filter(lambda x: x != "", paths))
    path = os.path.join(*paths)
    path = os.path.normpath(path)
    return path


def save_json(d, path, indent=1):
    with open(path, "w") as f:
        json.dump(d, f, indent=indent)

