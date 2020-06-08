def generate_pathway_los_config():
    config_dict = {'pathway_los_continuous_features': ['patient_age'],
                   'pathway_los_categorical_features': ['patient_condition', 'patient_point_of_entry'],
                   'pathway_los_target': ['patient_pathway', 'ward_los'],
                   'max_embedding_dim': 50
                   }

    verify_config(config_dict)

    return config_dict


def verify_config(config_dict):
    """Ensure target contains both pathway and LoS """

    if not (set(config_dict['pathway_los_target']) == {'patient_pathway', 'ward_los'}):
        print("Target variables must contain 'patient_pathway' and 'ward_los' only!")
        return ValueError

    if (config_dict['pathway_los_continuous_features'] + config_dict['pathway_los_categorical_features']) is None:
        print("There must be at least one feature to train on!")
        return ValueError

    return 0
