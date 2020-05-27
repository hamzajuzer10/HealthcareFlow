"""
-- HOSPITAL SIMULATION --

Description:
1. Patients arrive in the hospital
2. They are given a random ward that they will move to once they get a free bed
3. Patients can move from ward to ward before leaving the sim
    We do this with a transition prob square matrix where columns/rows are wards + discharge
8. Develop the ED process
4. Random arrival time for patients [easy]

Future steps:
5. Random los_dist for each ward stay (log norm) [easy]
6. Assign other attributes to the patient class (e.g. age, condition, gender) that will define their los_dist and their ward
    transition matrix (also needs to take into account the number of wards already visited) [hard]
7. Add additional resources e.g. nurses, consultants, X-ray machine etc.
    These resources can be local to the ward or part of the entire ward system [medium-hard]
9. Develop the Discharge process [medium]
10. Edge case for patients swapping beds when they need each other's bed and there is no bed availability [very hard]
11. Add first ward in the transition matrix [easy]
12. Change code so that arrival is part of the transition matrix
"""
from simulation import simulation
from simulation_dataset import hospital_initialisation_dataset, sim_configuration_dataset, presim_test_dataset
import datetime
import os
import sys


# To silence the prints in the model when we try to find the optimal
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == "__main__":

    # generate the sim config dataset
    sim_config = sim_configuration_dataset.generate_sim_config()

    # generate the presimulation test dataset
    patient_group_df, ward_df, pathway_df, los_df, demand_data_df = \
        presim_test_dataset.generate_presim_data(initial_timestamp=sim_config['sim_initial_timestamp'],
                                                 end_timestamp=sim_config['sim_end_timestamp'],
                                                 arrival_rate=5)

    # generate the hospital initialisation state dataset
    ward_df = hospital_initialisation_dataset.generate_ward_capacity(ward_df=ward_df, ward_capacity={'001': 50,
                                                                                                     '002': 25,
                                                                                                     '003': 45,
                                                                                                     '004': 60,
                                                                                                     '005': 80,
                                                                                                     '006': 70,
                                                                                                     '007': 35,
                                                                                                     '008': 20,
                                                                                                     '009': sys.maxsize})



    if sim_config['hospital_init']:

        hospital_init_patient_group_df, hospital_init_pathway_df, hospital_init_los_df, hospital_init_demand_data_df = \
        hospital_initialisation_dataset.generate_init_state(initial_timestamp= sim_config['sim_initial_timestamp'],
                                                            bed_occupancy_rate= {'004': 0.54, '005': 0.6, '007': 1.0, '008': 1.0},
                                                            ward_queues= {'007': 5, '008': 6},
                                                            expected_ward_LoS= 120,
                                                            stdev_ward_LoS= 10,
                                                            ward_LoS_cap_min= 15,
                                                            ward_LoS_cap_max= 540,
                                                            ward_df=ward_df)
    else:
        hospital_init_patient_group_df, hospital_init_pathway_df, hospital_init_los_df, hospital_init_demand_data_df = None

    if sim_config['actual_vs_forecast'] == 'actual':

        demand_data_df = demand_data_df[demand_data_df['actual_vs_forecast'] == 'actual']

    else:

        demand_data_df = demand_data_df[demand_data_df['actual_vs_forecast'] == 'forecast']


    # Set parameters and run the model
    simulation.model_run(patient_group_df,
                         ward_df,
                         pathway_df,
                         los_df,
                         demand_data_df,
                         sim_config,
                         hospital_init_patient_group_df,
                         hospital_init_pathway_df,
                         hospital_init_los_df,
                         hospital_init_demand_data_df,
                         save=True)
