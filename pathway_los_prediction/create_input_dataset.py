import pandas as pd
import json
import os
from pathway_los_prediction import model_configuration
from collections import Counter


def load_test_data(path):
    df = pd.read_pickle(path)
    df.reset_index(drop=True, inplace=True)

    return df


def create_input_files(input_file_path, output_file_dir, max_pathway_len, min_ward_freq, config_dict: dict,
                       train_pr=0.8, val_pr=0.1, test_pr=0.1):
    """
        Creates input files for training, validation, and test data.

        :param input_file_path: name of input file path
        :param min_ward_freq: wards occuring less frequently than this threshold are binned as <unk>s
        :param output_file_dir: name of output file path
        :param max_pathway_len: don't sample pathways longer than this length
        :param config_dict: pathway_los configuration dict
        :param train_pr: train dataset proportion
        :param val_pr: val dataset proportion
        :param test_pr: test dataset proportion
        """

    data_df = load_test_data(input_file_path)

    # only take the cols containing the relevant features and target data
    data_df = data_df[
        config_dict['pathway_los_continuous_features'] + config_dict['pathway_los_categorical_features'] + config_dict[
            'pathway_los_target']]

    # get the number of wards in the pathway
    data_df['pathway_len'] = data_df.apply(lambda x: len(x['patient_pathway']), axis=1)

    # remove all data samples where pathway length is larger than max_pathway_len
    data_df = data_df[data_df['pathway_len'] <= max_pathway_len]

    # get a ward count for all wards in the pathways (based on full dataset)
    wards_list = data_df['patient_pathway'].tolist()
    wards_list = [y for x in wards_list for y in x]
    ward_freq = Counter(wards_list)

    # Create ward map
    wards = [w for w in ward_freq.keys() if ward_freq[w] > min_ward_freq]
    ward_map = {k: v + 1 for v, k in enumerate(wards)}
    ward_map['<unk>'] = len(ward_map) + 1
    ward_map['<start>'] = len(ward_map) + 1
    ward_map['<end>'] = len(ward_map) + 1
    ward_map['<pad>'] = 0

    def encode_ward(x):
        ward_list = x['patient_pathway']
        enc_p = [ward_map['<start>']] + [ward_map.get(ward, ward_map['<unk>']) for ward in ward_list] + \
                [ward_map['<end>']] + [ward_map['<pad>']] * (max_pathway_len - len(ward_list))

        return enc_p

    def encode_ward_los(x):
        ward_los_list = x['ward_los']
        enc_p = [0] + ward_los_list + \
                [0] + [0] * (max_pathway_len - len(ward_los_list))

        return enc_p

    # Encode wards
    data_df['patient_pathway'] = data_df.apply(encode_ward, axis=1)

    # Encode ward LoS
    data_df['ward_los'] = data_df.apply(encode_ward_los, axis=1)

    # update pathway length to include both start and end codes
    data_df['pathway_len'] += 2

    # split data into train, val and test
    train_pr = train_pr / (train_pr + val_pr + test_pr)
    val_pr = val_pr / (train_pr + val_pr + test_pr)
    test_pr = test_pr / (train_pr + val_pr + test_pr)
    train_df = data_df.sample(frac=train_pr, random_state=200)  # random state is a seed value
    _df = data_df.drop(train_df.index)
    val_df = _df.sample(frac=val_pr / (val_pr + test_pr), random_state=200)  # random state is a seed value
    test_df = _df.drop(val_df.index)

    # reset index
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Create a feature map for all categorical features (based on train dataset)
    feature_map = {}
    for col in train_df[config_dict['pathway_los_categorical_features']]:
        v_unique = train_df[col].unique()
        map_name = col + "_map"
        _feature_map = {k: v + 1 for v, k in enumerate(v_unique)}
        feature_map[map_name] = _feature_map

    # apply categorical feature maps to train, val and test datasets
    for col in train_df[config_dict['pathway_los_categorical_features']]:
        train_df[col] = train_df[col].map(feature_map[str(col) + "_map"])
    train_df[col].fillna(value=0, inplace=True)

    for col in val_df[config_dict['pathway_los_categorical_features']]:
        val_df[col] = val_df[col].map(feature_map[str(col) + "_map"])
    val_df[col].fillna(value=0, inplace=True)

    for col in test_df[config_dict['pathway_los_categorical_features']]:
        test_df[col] = test_df[col].map(feature_map[str(col) + "_map"])
    test_df[col].fillna(value=0, inplace=True)

    # Create a map of categorical features to embedding dimensions
    cat_dims = [int(train_df[col].nunique()) for col in config_dict['pathway_los_categorical_features']]
    emb_dims = [[x, min(config_dict['max_embedding_dim'], (x + 1) // 2)] for x in cat_dims]

    # Save ward map, feature maps to a json
    with open(os.path.join(output_file_dir, 'WARD_MAP' + '.json'), 'w') as j:
        json.dump(ward_map, j)

    with open(os.path.join(output_file_dir, 'FEATURE_MAP' + '.json'), 'w') as j:
        json.dump(feature_map, j)

    # Save encoded pathways and their lengths to JSON files
    with open(os.path.join(output_file_dir, 'TRAIN' + '_PATHWAYS' + '.json'), 'w') as j:
        json.dump(train_df['patient_pathway'].tolist(), j)

    with open(os.path.join(output_file_dir, 'VAL' + '_PATHWAYS' + '.json'), 'w') as j:
        json.dump(val_df['patient_pathway'].tolist(), j)

    with open(os.path.join(output_file_dir, 'TEST' + '_PATHWAYS' + '.json'), 'w') as j:
        json.dump(test_df['patient_pathway'].tolist(), j)

    with open(os.path.join(output_file_dir, 'TRAIN' + '_PATHWAY_LEN' + '.json'), 'w') as j:
        json.dump(train_df['pathway_len'].tolist(), j)

    with open(os.path.join(output_file_dir, 'VAL' + '_PATHWAY_LEN' + '.json'), 'w') as j:
        json.dump(val_df['pathway_len'].tolist(), j)

    with open(os.path.join(output_file_dir, 'TEST' + '_PATHWAY_LEN' + '.json'), 'w') as j:
        json.dump(test_df['pathway_len'].tolist(), j)

    # Save ward LoS to a json
    with open(os.path.join(output_file_dir, 'TRAIN' + '_LOS' + '.json'), 'w') as j:
        json.dump(train_df['ward_los'].tolist(), j)

    with open(os.path.join(output_file_dir, 'VAL' + '_LOS' + '.json'), 'w') as j:
        json.dump(val_df['ward_los'].tolist(), j)

    with open(os.path.join(output_file_dir, 'TEST' + '_LOS' + '.json'), 'w') as j:
        json.dump(test_df['ward_los'].tolist(), j)

    # Save continuous features to a json
    with open(os.path.join(output_file_dir, 'TRAIN' + '_CONT_FEATURES' + '.json'), 'w') as j:
        json.dump(train_df[config_dict['pathway_los_continuous_features']].values.tolist(), j)

    with open(os.path.join(output_file_dir, 'VAL' + '_CONT_FEATURES' + '.json'), 'w') as j:
        json.dump(val_df[config_dict['pathway_los_continuous_features']].values.tolist(), j)

    with open(os.path.join(output_file_dir, 'TEST' + '_CONT_FEATURES' + '.json'), 'w') as j:
        json.dump(test_df[config_dict['pathway_los_continuous_features']].values.tolist(), j)

    # Save categorical features to a json
    with open(os.path.join(output_file_dir, 'TRAIN' + '_CAT_FEATURES' + '.json'), 'w') as j:
        json.dump(train_df[config_dict['pathway_los_categorical_features']].values.tolist(), j)

    with open(os.path.join(output_file_dir, 'VAL' + '_CAT_FEATURES' + '.json'), 'w') as j:
        json.dump(val_df[config_dict['pathway_los_categorical_features']].values.tolist(), j)

    with open(os.path.join(output_file_dir, 'TEST' + '_CAT_FEATURES' + '.json'), 'w') as j:
        json.dump(test_df[config_dict['pathway_los_categorical_features']].values.tolist(), j)

    # Save embedding dims to json
    with open(os.path.join(output_file_dir, 'EMB_DIMS' + '.json'), 'w') as j:
        json.dump(emb_dims, j)

if __name__ == "__main__":

    create_input_files(input_file_path='..\\sample_datasets\\sample_patient_pathway_los_data.pkl',
                       output_file_dir='..\\sample_datasets',
                       max_pathway_len=6,
                       min_ward_freq=20,
                       config_dict=model_configuration.generate_pathway_los_config())
