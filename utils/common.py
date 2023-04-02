import yaml 
import time

def read_yaml(filePath : object) -> str:
    with open(filePath, encoding="utf8") as f:
        content = yaml.safe_load(f)
    return content

def get_timestamp(file_name: str) -> str:
    timestamp = time.asctime().replace(" ", "_").replace(":", ".")
    unique_name = f"{file_name}_at_{timestamp}"
    return unique_name