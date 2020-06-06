def generate_pathway_los_config():

    config_dict = {'pathway_los_continuous_features': ['patient_age'],
                   'pathway_los_categorical_features': ['patient_condition', 'patient_point_of_entry'],
                   'pathway_los_target': ['patient_pathway'],
                   'minmax_scale_continuous_features': True
    }

    verify_config(config_dict)

    return config_dict


def verify_config(config_dict):
    """Ensure target contains either pathway only or pathway and LoS """

    if ~set(config_dict['pathway_los_target']).issubset(['patient_pathway', 'ward_los']) or ~{'patient_pathway'}.issubset(config_dict['pathway_los_target']):
        return ValueError

    return 0

