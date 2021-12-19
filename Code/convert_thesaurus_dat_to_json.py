import json
import re
from typing import List

import numpy as np


def read_dat(path: str) -> List[str]:
    with open(path, 'r') as file:
        return file.readlines()[1:]


# def read_dat(path: str) -> np.array:
#     return np.fromfile(path, dtype="byte")


if __name__ == '__main__':

    final_json = {}
    all_lines = read_dat('th_en_US_new.dat')

    current_key: str = ''
    for line in all_lines:
        if matches := re.findall(r'^\'?([^|]*)', line):
            if not (matches[0].startswith('(') and matches[0].endswith(')')):
                current_key = matches[0].strip().lower()
                final_json[current_key] = []

            elif current_key and '(noun)' in matches[0]:
                final_json[current_key] += line.strip().split('|')[1:]

    final_json = {key: values for key, values in final_json.items() if values}

    with open('thesaurus.json', 'w') as file:
        json.dump(final_json, file, indent=4)

