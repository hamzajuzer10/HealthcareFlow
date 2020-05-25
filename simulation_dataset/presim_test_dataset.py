import pandas as pd
import numpy as np
import datetime as datetime
import random as random
from simulation_dataset import los_distribution

def generate_demand(patient_groups, initial_timestamp: datetime.datetime,
                    end_timestamp: datetime.datetime, arrival_rate, actual_vs_forecasted):

    arrival_time = initial_timestamp
    patient_id = 1

    # create a final dataframe object
    results_df = pd.DataFrame()

    while arrival_time < end_timestamp:

        # sample from patient group list
        patient_group = random.choice(patient_groups)

        # create a dataframe
        data = {'patient_id': [patient_id],
                'patient_group': [patient_group],
                'timestamp': [arrival_time],
                'actual_vs_forecast':[actual_vs_forecasted]
                }
        data_df = pd.DataFrame(data,
                               columns=['patient_id', 'patient_group', 'timestamp', 'actual_vs_forecast'])

        # use an exponential process to generate timestamp data
        arrival_time += datetime.timedelta(minutes=round(np.random.exponential(arrival_rate), 0))
        patient_id += 1

        # concat with results_df
        results_df = pd.concat([results_df, data_df], axis=0)

    return results_df


def generate_presim_data(initial_timestamp: datetime.datetime, end_timestamp: datetime.datetime):
    # create test datasets to feed into the simulation
    patient_group_data = {'patient_group':  ['A', 'B', 'C', 'D'],
                          'mean_age': [25, 40, 50, 65],
                          'predominant_condition': ['diabetes', 'heart_illness', 'pneumonia', 'cancer'],
                          'predomimant_gender': ['male', 'female', 'male', 'female']
                          }

    patient_group_df = pd.DataFrame(patient_group_data, columns=['patient_group', 'mean_age', 'predominant_condition', 'predomimant_gender'])

    ward_data = {'ward_id': ['001', '002', '003', '004', '005', '006', '007', '008', '009'],
                 'ward_name': ['A_and_E', 'Elective', 'Ward_1', 'Ward_2', 'Ward_3', 'Ward_4', 'Ward_5', 'Ward_6',
                               'Sink_Ward'],
                 'opening_times': [datetime.time(7, 0, 0), datetime.time(7, 0, 0), datetime.time(7, 0, 0), datetime.time(9, 0, 0),
                                   datetime.time(9, 0, 0), datetime.time(13, 0, 0), datetime.time(9, 0, 0), datetime.time(8, 0, 0),
                                   datetime.time(8, 0, 0)],
                 'closing_times': [datetime.time(17, 0, 0), datetime.time(17, 0, 0), datetime.time(17, 0, 0), datetime.time(17, 0, 0),
                                   datetime.time(17, 0, 0), datetime.time(16, 0, 0), datetime.time(17, 0, 0), datetime.time(16, 0, 0),
                                   datetime.time(21, 0, 0)],
                 'ward_type': ['short_stay', 'admissions_lounge', 'long_stay', 'long_stay', 'long_stay', 'long_stay', 'long_stay',
                               'long_stay', 'sink'],
                 'capacity_limit': [True, True, True, True, True, True, True, True, True]}

    ward_df = pd.DataFrame(ward_data, columns=['ward_id', 'ward_name', 'opening_times', 'closing_times', 'ward_type', 'capacity_limit'])

    pathway_data = {'patient_group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D', 'D'],
                    'pathway': [['001', '003', '005', '006'],
                                ['001', '003', '004', '008'],
                                ['001', '003', '006'],
                                ['002', '006', '008'],
                                ['002', '006'],
                                ['002', '006', '008'],
                                ['002', '008'],
                                ['002', '004', '008'],
                                ['002', '004', '005'],
                                ['001', '003'],
                                ['002', '005', '006'],
                                ['001', '007'],
                                ['001', '007', '008'],
                                ['001', '004', '008'],
                                ['001', '004', '006'],
                                ['001', '003', '006']],
                    'pathway_probability': [0.3, 0.3, 0.2, 0.2, 0.5, 0.1, 0.1, 0.25, 0.05, 0.3, 0.5, 0.2, 0.9, 0.05, 0.025, 0.025]}

    pathway_df = pd.DataFrame(pathway_data, columns=['patient_group', 'pathway', 'pathway_probability'])

    los_data = {'patient_group': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                                  'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
                                  'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                  'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
                'ward_id': ['001', '002', '003', '004', '005', '006', '007', '008', '009',
                            '001', '002', '003', '004', '005', '006', '007', '008', '009',
                            '001', '002', '003', '004', '005', '006', '007', '008', '009',
                            '001', '002', '003', '004', '005', '006', '007', '008', '009'],
                'los_probability': [los_distribution.LoSDistribution(120, 120, 0, 20000), los_distribution.LoSDistribution(45, 30, 0, 20000), los_distribution.LoSDistribution(1300, 1000, 0, 20000),
                                    los_distribution.LoSDistribution(5000, 1500, 0, 20000), los_distribution.LoSDistribution(3500, 850, 0, 20000), los_distribution.LoSDistribution(1250, 250, 0, 20000),
                                    los_distribution.LoSDistribution(10000, 2500, 0, 20000), los_distribution.LoSDistribution(8000, 2000, 0, 20000), los_distribution.LoSDistribution(1500, 150, 0, 20000),
                                    los_distribution.LoSDistribution(120, 120, 0, 20000), los_distribution.LoSDistribution(45, 30, 0, 20000),
                                    los_distribution.LoSDistribution(6000, 3000, 0, 20000), los_distribution.LoSDistribution(3500, 300, 0, 20000), los_distribution.LoSDistribution(4500, 1000, 0, 20000),
                                    los_distribution.LoSDistribution(12000, 2000, 0, 20000), los_distribution.LoSDistribution(5600, 450, 0, 20000), los_distribution.LoSDistribution(8000, 300, 0, 20000),
                                    los_distribution.LoSDistribution(7500, 1300, 0, 20000), los_distribution.LoSDistribution(120, 120, 0, 20000),
                                    los_distribution.LoSDistribution(45, 30, 0, 20000), los_distribution.LoSDistribution(8000, 3000, 0, 20000), los_distribution.LoSDistribution(5600, 1200, 0, 20000),
                                    los_distribution.LoSDistribution(1200, 560, 0, 20000), los_distribution.LoSDistribution(500, 150, 0, 20000), los_distribution.LoSDistribution(5600, 1000, 0, 20000),
                                    los_distribution.LoSDistribution(9500, 1000, 0, 20000), los_distribution.LoSDistribution(3600, 1200, 0, 20000),
                                    los_distribution.LoSDistribution(120, 120, 0, 20000), los_distribution.LoSDistribution(45, 30, 0, 20000), los_distribution.LoSDistribution(3500, 300, 0, 20000),
                                    los_distribution.LoSDistribution(4500, 3000, 0, 20000), los_distribution.LoSDistribution(1200, 300, 0, 20000), los_distribution.LoSDistribution(15000, 10000, 0, 20000),
                                    los_distribution.LoSDistribution(4500, 1500, 0, 20000), los_distribution.LoSDistribution(8900, 900, 0, 20000), los_distribution.LoSDistribution(300, 120, 0, 20000)]}

    los_df = pd.DataFrame(los_data, columns=['patient_group', 'ward_id', 'los_probability'])

    demand_data_df_actual = generate_demand(['A', 'B', 'C', 'D'], initial_timestamp,
                                            end_timestamp, 30, 'actual')

    demand_data_df_forecast = generate_demand(['A', 'B', 'C', 'D'], initial_timestamp,
                                            end_timestamp, 30, 'forecast')

    demand_data_df = pd.concat([demand_data_df_actual, demand_data_df_forecast], axis=0)

    return patient_group_df, ward_df, pathway_df, los_df, demand_data_df



