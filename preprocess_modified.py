"""
Modified preprocessing for new dataset. 
Model now attempts to predict the outcome of each math. 
"""
import random
from random import shuffle

SURFACES = ['"Clay"', '"Grass"', '"Hard"', '"Carpet"']

def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    """Element-wise addition for two vectors, non-mutating."""
    assert len(vec1) == len(vec2)
    return [e1+e2 for e1,e2 in zip(vec1, vec2)]

def random_label() -> int:
    """Returns either 0 for 1 with equal probability."""
    return random.randint(0, 1)

def get_mapping_colnameTOidx(data: str | list[str]):
    """
    Reads the first line of csv and intialize bidirectional mapping dictionaries.
    Accepts either filepath string or list of labels. 
    """
    if isinstance(data, str):
        with open(data, "r", encoding="utf-8") as f:
            info_line = f.readline().strip("\n").split(",")
    elif isinstance(data, list):
        info_line = data
    
    mapping_dict = {}
    for idx,colname in enumerate(info_line):
        mapping_dict[idx] = colname
        mapping_dict[colname] = idx
    return mapping_dict

def process_dataset(data_filepath, write_filepath=None):
    """ Encode each match data of the two finalists from each tourament as a list of vectors and match result. 
        Writes calculated vectors to file
        Encoded vector is a vector of floats, result is either 0 or 1. 0 If first person wins. 
    """
    mapping_dict = {}
    surface_mapping = get_mapping_colnameTOidx(SURFACES)
    tournament_encodings = []
    
    with open(data_filepath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip("\n").split(",")
            if line[0] == '"surface"': # Skip .csv column name row
                mapping_dict = get_mapping_colnameTOidx(line)
                continue
                
            line[0] = str(surface_mapping[line[0]])
            line, label= line[:-1], line[-1]
            encoding = line
            tournament_encodings.append(",".join(encoding)+f";{label}")

    shuffle(tournament_encodings)
    with open(write_filepath, "w", encoding="utf-8") as f: 
        for line in tournament_encodings:
            f.write(line+"\n")
    return


def train_test_split(in_filepath, out_train, out_test, split_ratio=0.8):
    with open(in_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    split_idx = round(len(lines)*split_ratio)
    train = lines[:split_idx]
    test = lines[split_idx:]
    with open(out_train, "w", encoding="utf-8") as f:
        f.writelines(train)
    with open(out_test, "w", encoding="utf-8") as f:
        f.writelines(test)


if __name__ == "__main__":
    
    FILEPATH = "wta_atp_modified_data.csv"
    OUTPUT = "modified_dataset"

    process_dataset(FILEPATH, OUTPUT) 

    # Train-test split
    TRAIN = "modified_train"
    TEST = "modified_test"
    TRAIN_TEST_SPLIT_RATIO = 0.8
    train_test_split(OUTPUT, TRAIN, TEST, TRAIN_TEST_SPLIT_RATIO)
