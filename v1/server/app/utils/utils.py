import json


def get_templates(file_path: str) -> dict:
    try:
        with open(file_path) as json_file:
            templates = json.load(json_file)
            return templates
    except:
        pass
