"""
Processes GitHub .csv format into a vector representing a finalist matchup.
Each tournament is processed in isolation, i.e. has no knowledge of prior or future tournaments. 
"""
import random

def add_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        vec1 = [0 for e in vec2]
    return [e1+e2 for e1,e2 in zip(vec1, vec2)]

def random_label():
    return random.randint(0, 1)

def get_mapping_colnameTOidx(data):
    """Reads the first line of csv and intialize bidirectional mapping dictionaries."""
    if isinstance(data, str):
        with open(data, "r", encoding="utf-8") as f:
            info_line = f.readline().split(",")
            info_line[-1] = info_line[-1].strip("\n") # Strip \n at end of each lin
    elif isinstance(data, list):
        info_line = data
    
    mapping_dict = {}
    for idx,colname in enumerate(info_line):
        mapping_dict[idx] = colname
        mapping_dict[colname] = idx
    return mapping_dict


def create_tournaments_dict(data_filepath):
    """Reads a data .csv, creates a tuple of (parsed_dict, mapping_dict)
    (
        dict[
            tourney_id: list[
                (tourny_match_list: list[str], winner_name: str, loser_name: str, round: str)
                ...
                ]
            ]
     ,   
        dict[
            colname_i: idx_i,
            idx_i: colname_i
        ]
    }

    Will drop unwanted columns, but does not processed NA fields.
    """

    mapping_dict = get_mapping_colnameTOidx(data_filepath)

    WANTED_COLs = ["winner_ht", "winner_age", "loser_ht", "loser_age", "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"]

    new_mapping_dict = get_mapping_colnameTOidx(WANTED_COLs)
    winner_name_idx = mapping_dict["winner_name"]
    loser_name_idx = mapping_dict["loser_name"]
    round_idx = mapping_dict["round"]

    tourney_dict = {}
    with open(data_filepath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            match_info = line.split(",")
            tourney_id = match_info[0]
            match_info[-1] = match_info[-1].strip("\n") # Strip \n at end of each line

            if tourney_id == "tourney_id":
                continue

            # Keep only the columns we want
            processed_line = [match_info[mapping_dict[colname]] for colname in WANTED_COLs]
            processed_match = (processed_line, match_info[winner_name_idx], match_info[loser_name_idx], match_info[round_idx])
            
            if tourney_id in tourney_dict:
                tourney_dict[tourney_id].append(processed_match)
            else:  
                tourney_dict[tourney_id] = [processed_match] 
                
    return (tourney_dict, new_mapping_dict)


def encode_tournament(tourney, mapping_dict):
    """ Encode each data from a tourament and result into a vector. 
        Returns: (encoded_vector, result)
        Encoded vector is a vector of floats, result is either 0 or 1. 0 If first person wins. 
    """
    match_encodings = []
    finalist_winner = None
    finalist_loser = None
    finalist_winner_encoding = []
    finalist_loser_encoding = []

    has_final = False
    for match in tourney: # Loop through tournament to find final winner and loser names
        match_info, winner_name, loser_name, round = match
        if round == "F":
            finalist_winner = winner_name
            finalist_loser = loser_name
            has_final = True

    if not has_final: # NOTE drop all tournaments without a final match
        return
 
    # Loop through again and calculate tourney winner and loser tournament encodings
    for match in tourney:
        match_info, winner_name, loser_name, round = match
        match_info = [float(element) if element else 0 for element in match_info]
        if finalist_winner == winner_name: #TODO Remove hard coded indices
            ht_diff = match_info[0] - match_info[2]
            age_diff = match_info[1] - match_info[3]
            ace_diff = match_info[5] - match_info[14]
            df_diff = match_info[6] - match_info[15]
            svtp_diff = match_info[7] - match_info[16]
            firstIn_diff = match_info[8] - match_info[17]
            firstWon_diff = match_info[9] - match_info[18]
            secondWon_diff = match_info[10] - match_info[19]
            SvGms_diff = match_info[11] - match_info[20]
            bpSaved_diff = match_info[12] - match_info[21]
            bpFaced_diff = match_info[13] - match_info[22]
            rank_diff = match_info[23] - match_info[25]
            rank_points_diff = match_info[24] - match_info[26]

            match_encoding = [ht_diff, age_diff, ace_diff, df_diff, svtp_diff, firstIn_diff, firstWon_diff, secondWon_diff, SvGms_diff, bpSaved_diff, bpFaced_diff, rank_diff, rank_points_diff]
            finalist_winner_encoding = add_vectors(finalist_winner_encoding, match_encoding) # NOTE pooling
        elif finalist_loser == winner_name:
            ht_diff = match_info[0] - match_info[2]
            age_diff = match_info[1] - match_info[3]
            ace_diff = match_info[5] - match_info[14]
            df_diff = match_info[6] - match_info[15]
            svtp_diff = match_info[7] - match_info[16]
            firstIn_diff = match_info[8] - match_info[17]
            firstWon_diff = match_info[9] - match_info[18]
            secondWon_diff = match_info[10] - match_info[19]
            SvGms_diff = match_info[11] - match_info[20]
            bpSaved_diff = match_info[12] - match_info[21]
            bpFaced_diff = match_info[13] - match_info[22]
            rank_diff = match_info[23] - match_info[25]
            rank_points_diff = match_info[24] - match_info[26]

            match_encoding = [ht_diff, age_diff, ace_diff, df_diff, svtp_diff, firstIn_diff, firstWon_diff, secondWon_diff, SvGms_diff, bpSaved_diff, bpFaced_diff, rank_diff, rank_points_diff]
            finalist_loser_encoding = add_vectors(finalist_loser_encoding, match_encoding) # NOTE pooling

        else:
            continue

    # Concatenating encodings into tournament encoding, assign binary label.
    minutes = match_info[4]
    label = random_label()
    if label == 0:
        tournament_vector = finalist_winner_encoding + finalist_loser_encoding + [minutes]
    else:
        tournament_vector = finalist_loser_encoding + finalist_winner_encoding + [minutes]
    encoding_string = ",".join([str(e) for e in tournament_vector]) + ";" + str(label)
    return encoding_string


def main(data_csv, save_filepath):
    """Create formatted data for a specific year/csv."""

    #print(get_mapping_colnameTOidx(FILEPATH))
    encoded_vectors = []
    tourney_dict, new_mapping_dict = create_tournaments_dict(data_csv)
    for tourney_id in tourney_dict.keys():
        encoded_vector = encode_tournament(tourney_dict[tourney_id], new_mapping_dict)
        if encoded_vector == None:
            continue
        encoded_vectors.append(encoded_vector)
    
    with open(save_filepath, "w", encoding="utf-8") as f:
        for line in encoded_vectors:
            f.write(line+"\n")

if __name__ == "__main__":
    FILEPATH = "TennisData/wta_2020.csv"
    main(FILEPATH, "test")