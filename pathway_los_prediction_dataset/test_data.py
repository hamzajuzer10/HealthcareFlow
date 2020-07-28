import pandas as pd
import numpy as np
import random as random
from simulation_dataset import los_distribution


def generate_patient_pathway_los(patient_groups, patient_group_characteristics, patient_pathways, ward_los,
                                 num_samples, num_steps, filename):

    patient_id = 1
    gen_steps = 0

    # create a final dataframe object
    results_df = pd.DataFrame()

    while patient_id < num_samples:

        # sample from patient group list
        patient_group = random.choice(patient_groups)

        # append condition
        condition = patient_group_characteristics[patient_group_characteristics['patient_group'] == patient_group].iloc[0]['condition']

        # append age
        age = round(max(18, min(np.random.normal(50, 10, 1)[0], 90)),0)

        # append pathways
        patient_pathway = patient_pathways[patient_pathways['patient_group'] == patient_group]
        pathway = patient_pathway.sample(weights=patient_pathway.pathway_probability).iloc[0]['pathway'].copy()

        # append LoS per pathway
        los = []
        for ward in pathway:

            # calculate LoS for ward
            ward_los_ = ward_los[ward_los['patient_group'] == patient_group]
            los_distribution = ward_los_[ward_los_['ward_name'] == ward].iloc[0]['los_probability']
            w_los = los_distribution.sample()
            los.append(round(w_los,0))

        # append patient characteristics

        # create a dataframe
        data = {'patient_id': [patient_id],
                'patient_condition': [condition],
                'patient_age': [age],
                'patient_point_of_entry': [pathway[0]],
                'patient_pathway': [pathway[1:]],
                'ward_los': [los[1:]]}
        data_df = pd.DataFrame(data,
                               columns=['patient_id', 'patient_condition', 'patient_age', 'patient_point_of_entry',
                                        'patient_pathway', 'ward_los'])

        # use an exponential process to generate timestamp data
        patient_id += 1

        # concat with results_df
        results_df = pd.concat([results_df, data_df], axis=0)

        if patient_id % num_steps == 0:

            print('Completed generating patients {a} to {b}'.format(a=(gen_steps*num_steps)+1,
                                                                    b=(gen_steps+1)*num_steps))
            gen_steps += 1

    save_path = filename + ".pkl"

    save_test_data(results_df, save_path)


def generate_test_data(filename, num_patients=100, num_steps=5000):

    # create test datasets to feed into the pathway and Los prediction
    patient_group_data = {'patient_group':  ['A', 'B', 'C', 'D'],
                          'condition': ['Diabetes', 'Heart', 'Kidney', 'Lungs']}

    patient_group_df = pd.DataFrame(patient_group_data, columns=['patient_group', 'condition'])

    pathway_data = {'patient_group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D', 'D'],
                    'pathway': [['A_and_E', 'Ward_1', 'Ward_3', 'Ward_4'],
                                ['A_and_E', 'Ward_1', 'Ward_2', 'Ward_6'],
                                ['A_and_E', 'Ward_1', 'Ward_4'],
                                ['Elective', 'Ward_4', 'Ward_6'],
                                ['Elective', 'Ward_4'],
                                ['Elective', 'Ward_4', 'Ward_6'],
                                ['Elective', 'Ward_6'],
                                ['Elective', 'Ward_2', 'Ward_6'],
                                ['Elective', 'Ward_2', 'Ward_3'],
                                ['A_and_E', 'Ward_1'],
                                ['Elective', 'Ward_3', 'Ward_4'],
                                ['A_and_E', 'Ward_5'],
                                ['A_and_E', 'Ward_5', 'Ward_6'],
                                ['A_and_E', 'Ward_2', 'Ward_6'],
                                ['A_and_E', 'Ward_2', 'Ward_4'],
                                ['A_and_E', 'Ward_1', 'Ward_4']],
                    'pathway_probability': [0.3, 0.3, 0.2, 0.2, 0.5, 0.1, 0.1, 0.25, 0.05, 0.3, 0.5, 0.2, 0.9, 0.05, 0.025, 0.025]}

    pathway_df = pd.DataFrame(pathway_data, columns=['patient_group', 'pathway', 'pathway_probability'])

    los_data = {'patient_group': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                                  'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
                                  'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                  'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
                'ward_name': ['A_and_E', 'Elective', 'Ward_1', 'Ward_2', 'Ward_3', 'Ward_4', 'Ward_5', 'Ward_6', 'Sink_Ward',
                              'A_and_E', 'Elective', 'Ward_1', 'Ward_2', 'Ward_3', 'Ward_4', 'Ward_5', 'Ward_6', 'Sink_Ward',
                              'A_and_E', 'Elective', 'Ward_1', 'Ward_2', 'Ward_3', 'Ward_4', 'Ward_5', 'Ward_6', 'Sink_Ward',
                              'A_and_E', 'Elective', 'Ward_1', 'Ward_2', 'Ward_3', 'Ward_4', 'Ward_5', 'Ward_6', 'Sink_Ward'],
                'los_probability': [los_distribution.LoSDistribution(120, 12, 0, 20000), los_distribution.LoSDistribution(45, 4.5, 0, 20000), los_distribution.LoSDistribution(1300, 130, 0, 20000),
                                    los_distribution.LoSDistribution(5000, 500, 0, 20000), los_distribution.LoSDistribution(3500, 350, 0, 20000), los_distribution.LoSDistribution(1250, 125, 0, 20000),
                                    los_distribution.LoSDistribution(10000, 1000, 0, 20000), los_distribution.LoSDistribution(8000, 800, 0, 20000), los_distribution.LoSDistribution(1500, 150, 0, 20000),
                                    los_distribution.LoSDistribution(120, 12, 0, 20000), los_distribution.LoSDistribution(45, 4.5, 0, 20000),
                                    los_distribution.LoSDistribution(6000, 600, 0, 20000), los_distribution.LoSDistribution(3500, 350, 0, 20000), los_distribution.LoSDistribution(4500, 450, 0, 20000),
                                    los_distribution.LoSDistribution(12000, 1200, 0, 20000), los_distribution.LoSDistribution(5600, 560, 0, 20000), los_distribution.LoSDistribution(8000, 800, 0, 20000),
                                    los_distribution.LoSDistribution(7500, 750, 0, 20000), los_distribution.LoSDistribution(120, 12, 0, 20000),
                                    los_distribution.LoSDistribution(45, 4.5, 0, 20000), los_distribution.LoSDistribution(8000, 800, 0, 20000), los_distribution.LoSDistribution(5600, 560, 0, 20000),
                                    los_distribution.LoSDistribution(1200, 120, 0, 20000), los_distribution.LoSDistribution(500, 50, 0, 20000), los_distribution.LoSDistribution(5600, 560, 0, 20000),
                                    los_distribution.LoSDistribution(9500, 950, 0, 20000), los_distribution.LoSDistribution(3600, 360, 0, 20000),
                                    los_distribution.LoSDistribution(120, 12, 0, 20000), los_distribution.LoSDistribution(45, 4.5, 0, 20000), los_distribution.LoSDistribution(3500, 350, 0, 20000),
                                    los_distribution.LoSDistribution(4500, 450, 0, 20000), los_distribution.LoSDistribution(1200, 120, 0, 20000), los_distribution.LoSDistribution(15000, 1500, 0, 20000),
                                    los_distribution.LoSDistribution(4500, 450, 0, 20000), los_distribution.LoSDistribution(8900, 890, 0, 20000), los_distribution.LoSDistribution(300, 30, 0, 20000)]}

    los_df = pd.DataFrame(los_data, columns=['patient_group', 'ward_name', 'los_probability'])

    generate_patient_pathway_los(['A', 'B', 'C', 'D'], patient_group_df, pathway_df, los_df, num_patients, num_steps, filename)


def save_test_data(df, path):

    df.to_pickle(path)


if __name__ == "__main__":

    generate_test_data(filename='..\\sample_datasets\\sample_patient_pathway_los_data', num_patients=50000)





