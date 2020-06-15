import datetime
import os
import sys
from collections import namedtuple
from functools import wraps
import pandas as pd
from simpy import Environment, Resource
from timebetween import is_time_between
import sim_configuration_dataset
from simulation_dataset import hospital_initialisation_dataset, presim_test_dataset


# global variable for event log
global_event_log = pd.DataFrame(columns=['patient_id', 'patient_group', 'init_patient', 'source', 'destination', 'time_stamp', 'event_type'])
# global variable for occupancy log
global_occupancy_log = pd.DataFrame(columns=['patient_id', 'patient_group', 'init_patient', 'state', 'start_time', 'end_time'])
# global resource log
global_resource_log = pd.DataFrame(columns=['ward_id', 'ward_name', 'timestamp', 'occupancy', 'occupancy_rate', 'queue'])
# global delay log
global_delay_log = pd.DataFrame(columns=['patient_id', 'patient_group', 'init_patient', 'source', 'destination', 'delays', 'timestamp'])
# total waiting time
T = 0


# To silence the prints in the model when we try to find the optimal
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Colour class to colour code simulation events with different colours in the command prompt
class Cols:
    # ARRIVAL & DISCHARGE
    blue = '\033[94m'
    # NON-DELAYED EVENTS
    green = '\033[32m'
    # DELAYED EVENTS
    red = '\033[31m'
    # REQUESTS
    magenta = "\033[95m"
    # Additional colours
    cyan = "\033[96m"
    yellow = '\033[33m'
    # DEMAND
    gray = "\033[90m"
    # back to standard cli format
    end = '\033[0m'


# Ward Class
class Ward:
    """
    Hospital Ward class
    """

    def __init__(self, env, ward_id, ward_name, ward_type, num_beds, opening_time, closing_time):
        self.ward_id = ward_id
        self.ward_name = ward_name
        self.ward_type = ward_type
        self.beds = Resource(env, capacity=num_beds)
        self.opening_time = opening_time
        self.closing_time = closing_time
        self.patch_resource(self.beds, post=self.monitor)  # Patches (only) this resource instance

    def monitor(self, resource):
        """This is our monitoring callback."""

        entry = {'ward_id': self.ward_id,
                 'ward_name': self.ward_name,
                 'timestamp': resource._env.hospital_now(),
                 'occupancy': resource.count,
                 'occupancy_rate': round(resource.count/self.beds.capacity,2),
                 'queue': len(resource.queue)}

        entry = pd.DataFrame([entry], columns=entry.keys())

        global global_resource_log

        global_resource_log = pd.concat([global_resource_log, entry], axis=0).reset_index(drop=True)

        print(Cols.gray
              + '[sim_time = %s]: '
              % resource._env.hospital_now().strftime("%m/%d/%Y, %H:%M:%S")
              + 'DEMAND - Ward %s has %d patients and %d in queue'
              % (self.ward_name, resource.count, len(resource.queue))
              + Cols.end)

    def patch_resource(self, resource, pre=None, post=None):
        """Patch *resource* so that it calls the callable *pre* before each
        put/get/request/release operation and the callable *post* after each
        operation.  The only argument to these functions is the resource
        instance.
        """

        def get_wrapper(func):
            # Generate a wrapper for put/get/request/release
            @wraps(func)
            def wrapper(*args, **kwargs):
                # This is the actual wrapper
                # Call "pre" callback
                if pre:
                    pre(resource)

                # Perform actual operation
                ret = func(*args, **kwargs)

                # Call "post" callback
                if post:
                    post(resource)

                return ret

            return wrapper

        # Replace the original operations with our wrapper
        for name in ['put', 'get', 'request', 'release']:
            if hasattr(resource, name):
                setattr(resource, name, get_wrapper(getattr(resource, name)))


# Multiple wards form a ward system
class WardSystem:
    """
    dictionary of multiple wards
    """

    def __init__(self, env, ward_df):
        self.wards = {}

        for _, row in ward_df.iterrows():
            self.wards[row['ward_id']] = Ward(env, row['ward_id'], row['ward_name'], row['ward_type'],
                                              row['bed_capacity'], row['opening_times'],
                                              row['closing_times'])


# Patient class
class Patient:
    """
    Patient class, containing the patient id and the patient admission/spell process
    """

    def __init__(self, env, patient_id, patient_group, patient_pathways, ward_los, arrival_time, initialising_patient=False):
        self.patient_id = patient_id
        self.patient_group = patient_group
        self.patient_pathways = patient_pathways
        self.ward_los = ward_los
        self.arrival_time = arrival_time
        self.initialising_patient = initialising_patient
        self.env = env

    def gen_delay_log(self, source, destination, total_delay, delays):
        """
        Create and log delays
        """
        global global_delay_log

        entry = {'patient_id': self.patient_id, 'patient_group': self.patient_group,
                 'init_patient': self.initialising_patient, 'source': source, 'destination': destination,
                 'total_delay': total_delay, 'delay_breakdown': delays, 'time_stamp': self.env.hospital_now()}
        entry = pd.DataFrame([entry], columns=entry.keys())
        global_delay_log = pd.concat([global_delay_log, entry], axis=0).reset_index(drop=True)

        return entry

    def gen_occupancy_log(self, state):
        """
        Create and log occupancy
        """

        global global_occupancy_log

        # find index of last row
        try:
            idx = global_occupancy_log.tail(1).index.item()

        except ValueError:
            idx = -1

        occ_log = {'patient_id': self.patient_id,
                   'patient_group': self.patient_group,
                   'init_patient': self.initialising_patient,
                   'state': state,
                   'start_time': self.env.hospital_now(),
                   'end_time': False}

        row = pd.Series(occ_log, name=idx+1)
        global_occupancy_log = global_occupancy_log.append(row)

        return idx+1

    def update_occupancy_log(self, index):
        """
        Update the end time of the occupancy log
        """

        global global_occupancy_log

        global_occupancy_log.loc[index, 'end_time'] = self.env.hospital_now()

    def gen_event_log(self, source, destination, event_type):
        """
        Create and log events
        """
        global global_event_log

        entry = {'patient_id': self.patient_id, 'patient_group': self.patient_group,
                 'init_patient': self.initialising_patient, 'source': source, 'destination': destination,
                 'time_stamp': self.env.hospital_now(), 'event_type': event_type}
        entry = pd.DataFrame([entry], columns=entry.keys())
        global_event_log = pd.concat([global_event_log, entry], axis=0).reset_index(drop=True)

        return entry

    def calc_admission_discharge_delay(self, new_ward, old_ward=None):
        """
        Calculate the delays in admission and discharge from a ward
        """

        time_mins = 0

        if old_ward:

            # Check the ward opening and closing times of the new_ward and the old ward
            # and ensure the current time falls within these times
            if not (is_time_between(self.env.hospital_now().time(), new_ward.opening_time, new_ward.closing_time) &
                    is_time_between(self.env.hospital_now().time(), old_ward.opening_time, old_ward.closing_time)):

                # calculate the time difference between now and whichever is larger of the
                # new ward or old ward opening times
                if datetime.datetime.combine(datetime.date.today(), new_ward.opening_time) > \
                   datetime.datetime.combine(datetime.date.today(), old_ward.opening_time):

                    l_opening_time = new_ward.opening_time

                else:

                    l_opening_time = old_ward.opening_time

                time_diff = datetime.datetime.combine(datetime.date.today(), l_opening_time) - \
                            datetime.datetime.combine(datetime.date.today(), self.env.hospital_now().time())
                time_mins, _ = divmod(time_diff.seconds, 60) # need to check if no. days need to be included

        else:

            # Check the ward opening and closing times of the new ward and ensure the
            # current time falls within these times
            if not is_time_between(self.env.hospital_now().time(), new_ward.opening_time, new_ward.closing_time):

                # calculate the time difference between now and the new ward opening time
                time_diff = datetime.datetime.combine(datetime.date.today(), new_ward.opening_time) - \
                            datetime.datetime.combine(datetime.date.today(), self.env.hospital_now().time())
                time_mins, _ = divmod(time_diff.seconds, 60) # need to check if no. days need to be included

        return time_mins

    def spell(self, ward_system, arrival_mins):
        """
        Generating the entire patient admission process from arrival to discharge
        """

        # store all requests as a named tuple in the req_list list
        req_list = []
        Request = namedtuple('Request', ['ward', 'request'])

        # Get pathway based on probability distribution (each pathway is a list of ward_ids)
        pathway = self.patient_pathways.sample(weights=self.patient_pathways.pathway_probability).iloc[0]['pathway'].copy()

        # zero the counter
        counter = 0

        # wait t mins till arrival time
        yield self.env.timeout(arrival_mins)

        # log the admission time and track the index
        log_index_waiting = self.gen_occupancy_log(state='Admission')

        while True:

            # track the current time
            go_time = self.env.hospital_now()

            if counter > 0:
                """"
                If a patient needs to move into a new ward he needs to stay in his old ward 
                until a bed in the new ward becomes available. 

                This is the tricky bit:
                Edge case patient 1 on bed A requesting bed B patient 2 on bed B requesting bed A. 
                These patients need to swap beds at the same time

                We need to make use of the request, resource and process attributes methods to make this swap happen.
                Difficult to implement as we will have to reshuffle the queue for each resource - for now lets, have a timer 
                and if the patient has been waiting for a while, put him in a sink ward (unlimited capacity)
                """
                pos = old_ward.ward_name

                print(Cols.magenta
                      + '[sim_time = %s]: ' % self.env.hospital_now().strftime("%m/%d/%Y, %H:%M:%S")
                      + 'REQUEST - Patient %s is currently in %s and needs to be transferred to %s'
                      % (self.patient_id, old_ward.ward_name, ward.ward_name)
                      + Cols.end)

                ev_type = 'request'
                src = old_ward.ward_id
                dst = ward.ward_id
                self.gen_event_log(source=src, destination=dst, event_type=ev_type)

                # calculate admission delay
                adm_d_delay = self.calc_admission_discharge_delay(ward, req_list[-1].ward)
                yield self.env.timeout(adm_d_delay)

                # append the ward and request
                req_list.append(Request(ward=ward, request=ward.beds.request()))

                yield req_list[-1].request
                yield old_ward.beds.release(req_list[-2].request)

            else:

                pos = 'Admission'
                src = 'Admission'

                # Request the first ward in the pathway
                ward = ward_system.wards[pathway.pop(0)]

                # Print the request statement
                print(Cols.magenta
                      + '[sim_time = %s]: ' % self.env.hospital_now().strftime("%m/%d/%Y, %H:%M:%S")
                      + 'REQUEST - Patient %s needs to be admitted into %s'
                      % (self.patient_id, ward.ward_name)
                      + Cols.end)

                ev_type = 'request'
                dst = ward.ward_id
                self.gen_event_log(source=pos, destination=dst, event_type=ev_type)

                # append the ward and request
                req_list.append(Request(ward=ward, request=ward.beds.request()))

                # calculate admission delay
                adm_d_delay = self.calc_admission_discharge_delay(req_list[-1].ward)

                yield self.env.timeout(adm_d_delay)
                yield req_list[-1].request

            # colour-code red when there is a delay for the patient to get a bed
            delay = self.env.hospital_now() - go_time
            delay_mins, _ = divmod(delay.seconds, 60)
            delay_mins += delay.days*24*60

            global T
            T = T + delay_mins

            # log delay
            dst = ward.ward_id
            self.gen_delay_log(src, dst, delay_mins, {'total_delay': delay_mins,
                                                      'admission_discharge_delay': adm_d_delay})

            col = Cols.red if delay_mins != 0 else Cols.green

            print(col
                  + '[sim_time = %s]: ' % self.env.hospital_now().strftime("%m/%d/%Y, %H:%M:%S")
                  + 'EVENT - Patient %s transferred from %s to %s, action was delayed by %d mins'
                  % (self.patient_id, pos, ward.ward_name, delay_mins)
                  + Cols.end)

            # add event to event log
            ev_type = 'move'
            self.gen_event_log(source=src, destination=dst, event_type=ev_type)

            # Stay in the Ward based on your w_los input parameters in mins
            los_distribution = self.ward_los[self.ward_los['ward_id'] == ward.ward_id].iloc[0]['los_probability']
            w_los = los_distribution.sample()

            log_index_ward = self.gen_occupancy_log(state=ward.ward_id)
            yield self.env.timeout(w_los)

            self.update_occupancy_log(index=log_index_ward)

            # Choose the patient's next ward in the pathway
            try:
                next_ward = ward_system.wards[pathway.pop(0)]

                old_ward = ward
                ward = next_ward

            except IndexError:

                # Reached the end of the pathway, patient is leaving the hospital
                ward.beds.release(req_list[-1].request)
                print(Cols.blue
                      + '[sim_time = %s]: ' % self.env.hospital_now().strftime("%m/%d/%Y, %H:%M:%S")
                      + 'DISCHARGE - Patient %s has been discharged'
                      % (self.patient_id)
                      + Cols.end)
                ev_type = 'discharge'
                src = ward.ward_id
                dst = 'discharge'
                self.gen_event_log(source=src, destination=dst, event_type=ev_type)
                self.update_occupancy_log(index=log_index_waiting)
                break


            counter += 1


def save_logs(path, dict):
    """Save dataframe objects"""

    for key, value in dict.items():

        # save global variables to a file
        value.to_csv(path + '{a}.txt'.format(a=key), encoding='utf-8')
        value.to_csv(path + '{a}.csv'.format(a=key), encoding='utf-8')


def model_run(ward_df,
              pathway_df,
              los_df,
              demand_data_df,
              sim_config,
              hospital_init_pathway_df=None,
              hospital_init_los_df=None,
              hospital_init_demand_data_df=None,
              save_dir=None):
    """
    Run the model
    """

    # Dynamically add initial timestamp and hospital now method to the simpy Env
    Environment.hospital_initial_timestamp = sim_config['sim_initial_timestamp']

    def hospital_now(self):
        n_now = self.hospital_initial_timestamp + datetime.timedelta(minutes=self.now)
        return n_now

    Environment.hospital_now = hospital_now

    temp_env = Environment()

    # reset the event log on every run
    global global_event_log
    global_event_log = pd.DataFrame(columns=['patient_id', 'patient_group', 'init_patient', 'source', 'destination', 'time_stamp', 'event_type'])
    global global_occupancy_log
    global_occupancy_log = pd.DataFrame(columns=['patient_id', 'patient_group', 'init_patient', 'state', 'start_time', 'end_time'])
    global global_resource_log
    global_resource_log = pd.DataFrame(columns=['ward_id', 'ward_name', 'timestamp', 'occupancy', 'occupancy_rate', 'queue'])
    global global_delay_log
    global_delay_log = pd.DataFrame(columns=['patient_id', 'patient_group', 'init_patient', 'source', 'destination', 'total_delay', 'delay_breakdown', 'timestamp'])
    global T
    T = 0

    # Create a ward system
    ward_system = WardSystem(temp_env, ward_df)

    # Calculate the time diff
    time_diff = sim_config['sim_end_timestamp'] - sim_config['sim_initial_timestamp']
    time_mins, _ = divmod(time_diff.seconds, 60)
    time_mins += time_diff.days*24*60

    # Run the hospital initialisation or warm up
    proc_list = []

    if sim_config['hospital_init']:

        # loop through hospital_init_demand_data_df
        for index, row in hospital_init_demand_data_df.iterrows():

            # calculate arrival delay
            arrival_diff = row['timestamp'] - sim_config['sim_initial_timestamp']
            arrival_mins, _ = divmod(arrival_diff.seconds, 60)
            arrival_mins += arrival_diff.days*24*60

            # Generate a process for each patient
            proc_list.append(temp_env.process(Patient(temp_env,
                                                      row['patient_id'],
                                                      row['patient_group'],
                                                      hospital_init_pathway_df[hospital_init_pathway_df['patient_group'] == row['patient_group']],
                                                      hospital_init_los_df[hospital_init_los_df['patient_group'] == row['patient_group']],
                                                      row['timestamp'],
                                                      initialising_patient=True).spell(ward_system, arrival_mins)))

    # Loop through the demand df
    for index, row in demand_data_df.iterrows():

        # calculate arrival delay
        arrival_diff = row['timestamp'] - sim_config['sim_initial_timestamp']
        arrival_mins, _ = divmod(arrival_diff.seconds, 60)
        arrival_mins += arrival_diff.days*24*60

        # Generate a process for each patient
        proc_list.append(temp_env.process(Patient(temp_env,
                                                  row['patient_id'],
                                                  row['patient_group'],
                                                  pathway_df[
                                                  pathway_df['patient_group'] == row[
                                                          'patient_group']],
                                                  los_df[los_df['patient_group'] == row[
                                                      'patient_group']],
                                                  row['timestamp'],
                                                  initialising_patient=False).spell(ward_system, arrival_mins)))

    temp_env.run(until=time_mins)

    # save logs
    if save_dir:
        save_dict = {'resource_log': global_resource_log,
                     'occupancy_log': global_occupancy_log,
                     'event_log': global_event_log,
                     'delay_log': global_delay_log,
                     'ward_log': ward_df}
        save_logs(path=save_dir, dict=save_dict)


if __name__ == "__main__":

    # generate the sim config dataset
    sim_config = sim_configuration_dataset.generate_sim_config()

    # generate the presimulation test dataset
    patient_group_df, ward_df, pathway_df, los_df, demand_data_df = \
        presim_test_dataset.generate_presim_data(initial_timestamp=sim_config['sim_initial_timestamp'],
                                                 end_timestamp=sim_config['sim_end_timestamp'],
                                                 arrival_rate=15)

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
        hospital_init_patient_group_df, hospital_init_pathway_df, hospital_init_los_df, hospital_init_demand_data_df = \
            None, None, None, None

    if sim_config['actual_vs_forecast'] == 'actual':

        demand_data_df = demand_data_df[demand_data_df['actual_vs_forecast'] == 'actual']

    else:

        demand_data_df = demand_data_df[demand_data_df['actual_vs_forecast'] == 'forecast']

    # Set parameters and run the model
    model_run(ward_df=ward_df,
              pathway_df=pathway_df,
              los_df=los_df,
              demand_data_df=demand_data_df,
              sim_config=sim_config,
              hospital_init_pathway_df=hospital_init_pathway_df,
              hospital_init_los_df=hospital_init_los_df,
              hospital_init_demand_data_df=hospital_init_demand_data_df,
              save_dir="..//logs//")
