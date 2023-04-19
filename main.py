from preprocess_modified import run_preprocessing
from baseline_logistic import run_stats_models
from baseline_ffn import run_ffn

def main():
    FILEPATH = "wta_atp_modified_data.csv"
    OUTPUT = "modified_dataset"

    # Train-test split
    TRAIN = "modified_train"
    TEST = "modified_test"
    TRAIN_TEST_SPLIT_RATIO = 0.8
    
    run_preprocessing(FILEPATH, OUTPUT, TRAIN, TEST, TRAIN_TEST_SPLIT_RATIO)
    run_stats_models(TRAIN, TEST)
    run_ffn(TRAIN, TEST)

if __name__ == "__main__":
    main()