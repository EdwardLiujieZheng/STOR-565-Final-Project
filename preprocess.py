"""
Processes GitHub .csv format into a vector representing a finalist matchup.
Each tournament is processed in isolation, i.e. has no knowledge of prior or future tournaments. 
"""
import random, os, sys

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


def create_tournaments_dict(data_filepath: str):
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
    winner_idx = mapping_dict["winner_name"]
    loser_idx = mapping_dict["loser_name"]
    round_idx = mapping_dict["round"]

    tourney_dict = {}
    with open(data_filepath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "":
                continue
            
            match_info = line.strip("\n").split(",")
            tourney_id = match_info[0] # First item is always tourney_id

            if tourney_id == "tourney_id": # NOTE skip first line
                continue

            winner = match_info[winner_idx]
            loser = match_info[loser_idx]

            processed_match = (match_info, winner, loser, match_info[round_idx])
            
            if tourney_id in tourney_dict:
                tourney_dict[tourney_id].append(processed_match)
            else:  
                tourney_dict[tourney_id] = [processed_match] 
                
    return (tourney_dict, mapping_dict)


def encode_tournament_vectors(tourneys, mapping_dict, write_filepath=None):
    """ Encode each match data of the two finalists from each tourament as a list of vectors and match result. 
        Writes calculated vectors to file
        Encoded vector is a vector of floats, result is either 0 or 1. 0 If first person wins. 
    """
    tournament_encodings = []
    for tourney_id in tourneys:
        tourney = tourneys[tourney_id] # list[(matches, winner_name, loser_name, round)]
        if len(tourney) == 1: # Only has final 
            continue

        finalist_winner = None
        finalist_loser = None
        finalist_winner_encodings = []
        finalist_loser_encodings = []

        has_final = False
        for match in tourney: # Loop through tournament to find final winner and loser names
            match_info, winner_name, loser_name, round = match
            if round == "F":
                finalist_winner = winner_name
                finalist_loser = loser_name
                has_final = True
                break
        if not has_final: # NOTE ignore all tournaments without a final match
            continue
    
        # Loop through again and calculate list of tourney winner and loser tournament encodings
        for match in tourney:
            match_info, winner_name, loser_name, round = match

            if round == "F": # Ignore final match
                continue

            if winner_name in (finalist_winner, finalist_loser):
                # Set all NA to 0, change str to float
                temp = []
                for data in match_info:
                    if not data:
                        temp.append(0.1)
                    else:
                        try:
                            temp.append(float(data))
                        except Exception:
                            temp.append(data)
                match_info = temp

                # NOTE Extract match features
                ht_diff = match_info[mapping_dict["winner_ht"]] - match_info[mapping_dict["loser_ht"]]
                age_diff = match_info[mapping_dict["winner_age"]] - match_info[mapping_dict["loser_age"]]
                ace_diff = match_info[mapping_dict["w_ace"]] - match_info[mapping_dict["l_ace"]]
                df_diff = match_info[mapping_dict["w_df"]] - match_info[mapping_dict["l_df"]]
                svpt_diff = match_info[mapping_dict["w_svpt"]] - match_info[mapping_dict["l_svpt"]]
                firstIn_diff = match_info[mapping_dict["w_1stIn"]] - match_info[mapping_dict["l_1stIn"]]
                firstWon_diff = match_info[mapping_dict["w_1stWon"]] - match_info[mapping_dict["l_1stWon"]]
                secondWon_diff = match_info[mapping_dict["w_2ndWon"]] - match_info[mapping_dict["l_2ndWon"]]
                svgms_diff = match_info[mapping_dict["w_SvGms"]] - match_info[mapping_dict["l_SvGms"]]
                bpSaved_diff = match_info[mapping_dict["w_bpSaved"]] - match_info[mapping_dict["l_bpSaved"]]
                bpFaced_diff = match_info[mapping_dict["w_bpFaced"]] - match_info[mapping_dict["l_bpFaced"]]
                rank_points_diff = match_info[mapping_dict["winner_rank_points"]] - match_info[mapping_dict["loser_rank_points"]]

                match_encoding = [ht_diff, age_diff, ace_diff, df_diff, svpt_diff, firstIn_diff, firstWon_diff, secondWon_diff, svgms_diff, bpSaved_diff, bpFaced_diff, rank_points_diff]

                if finalist_winner == winner_name:
                    finalist_winner_encodings.append(",".join([str(ele) for ele in match_encoding]))
                elif finalist_loser == winner_name:
                    finalist_loser_encodings.append(",".join([str(ele) for ele in match_encoding]))
                    
            else:
                continue # Ignore all matches that don't involve tournament finalists

        # Concatenating encodings into tournament encoding, assign binary label.
        label = random_label() # Decides order or contatenation
        assert len(finalist_winner_encodings) != 0
        assert len(finalist_loser_encodings) != 0
        
        if len(finalist_loser_encodings) != len(finalist_winner_encodings): # Missing matches
            continue

        if label == 0:
            tournament_vector = ";".join(finalist_winner_encodings) + "~" + ";".join(finalist_loser_encodings) + f"~{label}"  
        else:
            tournament_vector = " ;".join(finalist_loser_encodings) + "~" + ";".join(finalist_winner_encodings) + f"~{label}"    
        tournament_encodings.append(tournament_vector)

    with open(write_filepath, "w", encoding="utf-8") as f: 
        for line in tournament_encodings:
            f.write(line+"\n")
    return


def addition_pooling_tournament_vectors(data_filepath, save_filepath):
    """Reads file generated from encode_tournament_vectors(). Pools the match data from each finalist."""

    encoded_tournaments = []
    with open(data_filepath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            player1_pooled = None
            player2_pooled = None
            if line == "":
                continue

            player1_matches, player2_matches, result = line.split("~")
            player1_matches_sep = player1_matches.split(";")
            player2_matches_sep = player2_matches.split(";")

            # Pool player1
            for match in player1_matches_sep:
                match_features = [float(ele) for ele in match.split(",")]
                if player1_pooled == None:
                    player1_pooled = match_features
                else:
                    player1_pooled = add_vectors(player1_pooled, match_features)
                
            # Pool player2
            for match in player2_matches_sep:
                match_features = [float(ele) for ele in match.split(",")]
                if player2_pooled == None:
                    player2_pooled = match_features
                else:
                    player2_pooled = add_vectors(player2_pooled, match_features)
            
            player1_pooled = ",".join([str(ele) for ele in player1_pooled])
            player2_pooled = ",".join([str(ele) for ele in player2_pooled])

            tournament = player1_pooled + ";" + player2_pooled + ";" + result
            encoded_tournaments.append(tournament)
                
    with open(save_filepath, "w", encoding="utf-8") as f:
        for line in encoded_tournaments:
            f.write(line)


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
    
    #FILEPATH = "wta_modified_dataset.csv"
    FILEPATH = "originalData"
    tourney_dict, new_mapping_dict = create_tournaments_dict(FILEPATH)

    # For each tournament, filter out irrelevant matches, extract and encode interested matches
    encode_tournament_vectors(tourney_dict, new_mapping_dict, 'tournament_vectors') 

    # Pool together data from each tournament into "player1_vector;player2_vector;label"
    addition_pooling_tournament_vectors("tournament_vectors", "pooled_tournament_vectors")

    # Tran-test split
    DATA = "pooled_tournament_vectors"
    TRAIN = "pooled_train"
    TEST = "pooled_test"
    TRAIN_TEST_SPLIT = 0.8
    train_test_split(DATA, TRAIN, TEST, TRAIN_TEST_SPLIT)
