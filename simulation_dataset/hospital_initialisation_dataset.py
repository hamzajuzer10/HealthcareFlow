import pandas as pd
import datetime as datetime
from simulation_dataset import presim_test_dataset, los_distribution
import sys


def generate_init_demand(ward_demand, initial_timestamp: datetime.datetime):

    patient_id = 1

    # create a final dataframe object
    results_df = pd.DataFrame()

    # loop through each patient group in the ward demand dict
    for patient_group in ward_demand.keys():

        # get the ward_id and demand
        data = {'patient_id': list(range(patient_id, patient_id + ward_demand[patient_group]['demand'])),
                'patient_group': [patient_group]*ward_demand[patient_group]['demand'],
                'timestamp': [initial_timestamp]*ward_demand[patient_group]['demand'],
                'actual_vs_forecasted': 'actual'
                }

        patient_id += ward_demand[patient_group]['demand']

        data_df = pd.DataFrame(data,
                               columns=['patient_id', 'patient_group', 'timestamp', 'actual_vs_forecasted'])


        # concat with results_df
        results_df = pd.concat([results_df, data_df], axis=0)

    return results_df


def generate_init_state(initial_timestamp: datetime.datetime,
                        bed_occupancy_rate: dict,
                        ward_queues: dict,
                        expected_ward_LoS: float,
                        stdev_ward_LoS: float,
                        ward_LoS_cap_min: float,
                        ward_LoS_cap_max: float,
                        ward_df: pd.DataFrame):
    """
    Initialise the state of the hospital
    1) Simulation start time
    3) Bed occupancy rate in each ward
    4) Queues for each ward (if bed occupancy rate is > 100%)  
    
    Create a new patient group for each ward.
    The pathways for each patient group is the ward and then discharge.
    The LoS in the ward and discharge is provided by the expected_los argument

        """

    # First, we check that all wards in ward_queues are also in bed_occupancy_rate dictionary
    if not set(ward_queues.keys()).issubset(bed_occupancy_rate.keys()):
        raise ValueError

    # Secondly, we check that all for all wards in the ward_queues dict, their values in the bed_occupancy dict is 100%
    for ward in ward_queues.keys():
        if bed_occupancy_rate[ward] != 1:
            raise ValueError

    # create init patient groups for each init ward (these are not the actual patient groups)
    patient_group = []
    los_patient_group = []
    pathway = []
    pathway_prob = []
    los_ward = []
    los_probability = []
    ward_demand = {}

    for ward in bed_occupancy_rate.keys():

        patient_gr = ward +'_init_group'
        patient_group.append(patient_gr)
        pathway.append([ward])
        pathway_prob.append(1)
        los_patient_group.extend([patient_gr])
        los_ward.extend([ward])
        los_probability.extend([los_distribution.LoSDistribution(expected_ward_LoS, stdev_ward_LoS,
                                                                 ward_LoS_cap_min, ward_LoS_cap_max)])
        patient_demand = {'demand': int(bed_occupancy_rate[ward]*ward_df[ward_df['ward_id'] == ward].bed_capacity.values[0] +
                                    ward_queues.get(ward, 0)),
                          'ward_id': ward}

        ward_demand[patient_gr] = patient_demand

    # create patient groups for each initialisation ward
    patient_group_data = {'patient_group': patient_group}

    patient_group_df = pd.DataFrame(patient_group_data, columns=['patient_group'])

    # create pathways for each patient group - one pathway per patient group (that ward followed by discharge)
    pathway_data = {'patient_group': patient_group,
                    'pathway': pathway,
                    'pathway_probability': pathway_prob}

    pathway_df = pd.DataFrame(pathway_data, columns=['patient_group', 'pathway', 'pathway_probability'])

    # create los_dist for each patient group and ward
    los_data = {'patient_group': los_patient_group,
                'ward_id': los_ward,
                'los_probability': los_probability}

    los_df = pd.DataFrame(los_data,
                          columns=['patient_group', 'ward_id', 'los_probability', 'los_cap_min', 'los_cap_max'])

    # create a demand profile starting at initial timestamp
    demand_data_df = generate_init_demand(ward_demand, initial_timestamp)

    return patient_group_df, pathway_df, los_df, demand_data_df


def generate_ward_capacity(ward_df: pd.DataFrame, ward_capacity: dict):

    # assert that all wards in ward_capacity exist in ward_df
    if not set(ward_df.ward_id.unique()).issubset(ward_capacity.keys()):
        raise ValueError

    ward_df['bed_capacity'] = ward_df['ward_id'].map(ward_capacity)

    return ward_df













