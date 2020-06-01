"""
-- HOSPITAL SIMULATION --

Description:
1. Patients arrive in the hospital
2. They are given a random ward that they will move to once they get a free bed
3. Patients can move from ward to ward before leaving the sim
    We do this with a transition prob square matrix where columns/rows are wards + discharge

Future steps:
5. Random los_dist for each ward stay (log norm) [easy]
6. Assign other attributes to the patient class (e.g. age, condition, gender) that will define their los_dist and their ward
    transition matrix (also needs to take into account the number of wards already visited) [hard]
7. Add additional resources e.g. nurses, consultants, X-ray machine etc.
    These resources can be local to the ward or part of the entire ward system [medium-hard]
9. Develop the Discharge process [medium]
10. Edge case for patients swapping beds when they need each other's bed and there is no bed availability [very hard]
11. Add first ward in the transition matrix [easy]
12. Add all existing patients from a DataFrame at time 0 with their remaining LOS etc. [easy]

Done:
8. Develop the ED process
4. Random arrival time for patients [easy]

Notes:
a. __dict__ for a process object
    'env': <simpy.core.Environment object at 0x0000017ABD2E6FD0>, 'callbacks': None, '_value': None
    ,'_generator': <generator object Patient.spell at 0x0000017ABAB54D68>, '_target': None, '_ok': True

b. __dict__ for requests
    env': <simpy.core.Environment object at 0x0000017ABC658FD0>
    ,'callbacks': [<bound method BaseResource._trigger_get of
                   <simpy.resources.resource.Resource object at 0x0000017ABC658780>>]
    ,'_value': None, 'resource': <simpy.resources.resource.Resource object at 0x0000017ABC658780>
    ,'proc': <Process(spell) object at 0x17abc6583c8>, 'usage_since': 0, '_ok': True}

b. __dict__ for resource
    '_env': <simpy.core.Environment object at 0x0000026597173748>, '_capacity': 1, 'put_queue': [], 'get_queue': []
    ,'request': <bound method Request of <simpy.resources.resource.Resource object at 0x0000026597173668>>
    ,'release': <bound method Release of <simpy.resources.resource.Resource object at 0x0000026597173668>>
    ,'users': [], 'queue': []
"""
import simpy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from functools import partial, wraps

# global variable for event log
global_event_log = pd.DataFrame(columns=['patient_id', 'source', 'destination', 'time_stamp', 'event_type'])
# global_occupancy_log = pd.DataFrame(columns=['patient_id', 'state', 'start_time', 'end_time'])
global_occupancy_log = []
# total waiting time
T = 0

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
    # Checks
    gray = "\033[90m"
    # back to standard cli format
    end = '\033[0m'


# Placeholder for ED
class ED:
    """
    Emergency department class
    """
    def __init__(self, env, num_rec, num_nurses):
        # ED (Emergency department) system - do it later
        self.receptionist = simpy.Resource(env, capacity=num_rec)
        self.nurse = simpy.Resource(env, capacity=num_nurses)


# Ward Class, resources: beds, nurses, doctors etc.
class Ward:
    """
    Hospital Ward class
    """
    def __init__(self, env, name, num_beds):
        self.name = name
        self.beds = simpy.Resource(env, capacity=num_beds)
        self.data = []
        self.monitor = partial(self.monitor, self.data)
        self.patch_resource(self.beds, post=self.monitor)  # Patches (only) this resource instance

    def monitor(self, data, resource):
        """This is our monitoring callback."""
        item = (
            resource._env.now,  # The current simulation time
            resource.count,  # The number of users
            len(resource.queue),  # The number of queued processes
        )
        data.append(item)
        print(Cols.magenta
              + '[sim_time = %d]: '
              % resource._env.now
              + 'DEMAND - Ward %s has %d patients and %d in queue'
              % (self.name, resource.count, len(resource.queue))
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


# Multiple wards make form a ward system
class WardSystem:
    """
    dictionary of multiple wards
    """
    def __init__(self, env, ward_dict):
        self.wards = {}
        for key in ward_dict:
            self.wards[key] = Ward(env, key, ward_dict[key])


# Patient class
# Will update with patient characteristics that will define their pathway, but for now wa add the ward directly into the
# class
class Patient:
    """
    Patient class, containing the patient id and the patient admission/spell process
    """
    def __init__(self, env, patient_id, arrival_time):
        self.patient_id = patient_id
        self.arrival_time = arrival_time
        self.env = env
        
    def generate_transition_matrix(self):
        """
        Placeholder for Transition matrix based on patient characteristics
        :return:
        """

    def gen_event_log(self, source, destination, event_type):
        """
        Add event to the event log
        :param source:
        :param destination:
        :param event_type:
        :return:
        """
        entry = {'patient_id': self.patient_id, 'source': source, 'destination': destination
                 , 'time_stamp': self.env.now, 'event_type': event_type}
        entry = pd.DataFrame([entry], columns=entry.keys())
        global global_event_log
        global_event_log = pd.concat([global_event_log, entry], axis=0).reset_index(drop=True)
        return entry

    def resource_check(self, ward):
        """
        Used for debugging to check the status of a resource
        :param ward:
        :return:
        """
        print(Cols.gray
              + '[sim_time = %d]: ' % self.env.now
              + 'CHECKS - Patient %s, Ward %s Intermediate RESOURCE check: %d, %d'
              % (self.patient_id, ward.name, ward.beds.count, ward.beds.users)
              + Cols.end)

    def admission(self, ed, adm_type, rec_time, nurse_time):
        """
        Admissions can be elective (EL) e.g. scheduled operations or non-elective (NEL)
        that come via ED (A&E) (or ambulances etc???)
        TO DO: need to add these events to the event log
        :param ed:
        :param adm_type:
        :param rec_time:
        :param nurse_time:
        :return:
        """
        yield self.env.timeout(self.arrival_time)
        global global_occupancy_log

        print(Cols.blue
              + '[sim_time = %d]: ' % self.env.now
              + '%s ARRIVAL - Patient %s arrived'
              % (adm_type, self.patient_id)
              + Cols.end)
        if adm_type == 'NEL':
            occ_log = {'patient_id': self.patient_id, 'state': 'ED'
                       , 'start_time': self.env.now, 'end_time': False}
            global_occupancy_log.append(occ_log)
            log_index = len(global_occupancy_log)-1
            with ed.receptionist.request() as req:
                yield req
                yield self.env.timeout(rec_time)
            with ed.nurse.request() as req:
                yield req
                yield self.env.timeout(nurse_time)
            global_occupancy_log[log_index]['end_time'] = self.env.now
        elif adm_type == 'EL':
            adm_type = 'EL'

    def calc_ward_los(self, mean, var):
        """
        Placeholder for a function that calculates the los_dist at each ward based on the patient's attributes
        (and samples from a log norm)
        Need to make changes to the spell function when this is ready to call it's returned value on every timeout
        related to a ward stay
        """
        wlos = np.random.lognormal(mean, var)
        return wlos

    # add the ward as an external input in the spell function
    def spell(self, ward, ward_system, tr_matrix, ed, adm_type):
        """
        Generating the entire patient admission process from arrival to discharge
        :param ward:
        :param ward_system:
        :param tr_matrix:
        :param ed:
        :param adm_type:
        :return:
        """
        global global_occupancy_log
        # to get the index use len(global_occupancy_log)-1
        req_list = []
        counter = 0
        # admission is a subprocess of the spell process
        admission = self.env.process(self.admission(ed, adm_type=adm_type, rec_time=10, nurse_time=45))
        yield admission

        # Admission entry in occupancy log
        occ_log = {'patient_id': self.patient_id, 'state': 'Waiting'
                   , 'start_time': self.env.now, 'end_time': False}
        global_occupancy_log.append(occ_log)
        log_index_waiting = len(global_occupancy_log) - 1

        while True:
            # Choose the next ward based on the transition matrix
            go_time = self.env.now
            req_list.append(ward.beds.request())

            pos = old_ward.name if counter > 0 else 'Waiting'  # replace with NEL and EL
            print(Cols.magenta
                  + '[sim_time = %d]: ' % self.env.now
                  + 'REQUEST - Patient %s is currently in %s and needs a bed in ward %s'
                  % (self.patient_id, pos, ward.name)
                  + Cols.end)
            ev_type = 'request'
            dst = ward.name
            self.gen_event_log(source=pos, destination=dst, event_type=ev_type)
            # Check request value attributes

            if counter > 0:
                """"
                If a patient needs to move into a new ward he needs to stay in his old ward 
                until a bed in the new ward becomes available. 

                This is the tricky bit:
                Edge case patient 1 on bed A requesting bed B patient 2 on bed B requesting bed A. 
                These patients need to swap beds at the same time

                We need to make use of the request, resource and process attributes methods to make this swap happen
                """
                yield req_list[-1]
                yield old_ward.beds.release(req_list[-2])
            else:
                yield req_list[-1]

            # colour-code red when there is a delay for the patient to get a bed
            delay = self.env.now - go_time
            global T
            T = T + delay
            col = Cols.red if delay != 0 else Cols.green
            print(col
                  + '[sim_time = %d]: ' % self.env.now
                  + 'EVENT - Patient %s moved from %s to %s, action was delayed by %d'
                  % (self.patient_id, pos, ward.name, delay)
                  + Cols.end)

            # add event to event log
            ev_type = 'move'
            dst = ward.name
            self.gen_event_log(source=pos, destination=dst, event_type=ev_type)

            # Stay in the Ward based on your w_los input parameters in days
            w_los = self.calc_ward_los(0, 1)*1440
            occ_log = {'patient_id': self.patient_id, 'state': ward.name
                       , 'start_time': self.env.now, 'end_time': False}
            global_occupancy_log.append(occ_log)
            log_index_ward = len(global_occupancy_log) - 1
            yield self.env.timeout(w_los)
            global_occupancy_log[log_index_ward]['end_time'] = self.env.now

            # Choose the patient's new destination
            ward_index = list(ward_system.wards.keys()).index(ward.name)
            # normalise transition matrix
            probs = tr_matrix[ward_index, :] / tr_matrix[ward_index, :].sum()
            next_ward = str(np.random.choice(list(ward_system.wards.keys()), 1, p=probs)[0])

            if next_ward == 'Discharge':
                # If patient is being discharged release the bed and go
                # If the patient is moving to another ward then keep the bed until
                # there is an available bed in the other ward
                ward.beds.release(req_list[-1])
                print(Cols.blue
                      + '[sim_time = %d]: ' % self.env.now
                      + 'DISCHARGE - Patient %s left his bed in %s and got discharged'
                      % (self.patient_id, ward.name)
                      + Cols.end)
                ev_type = 'discharge'
                src = ward.name
                dst = 'discharge'
                self.gen_event_log(source=src, destination=dst, event_type=ev_type)
                global_occupancy_log[log_index_waiting]['end_time'] = self.env.now
                break

            else:
                old_ward = ward
                ward = ward_system.wards[next_ward]

            counter += 1


def gantt_plot(occupancy_log, input_wards, save_file=""):
    df_occupancy_log = pd.DataFrame(occupancy_log, columns=['patient_id', 'state', 'start_time', 'end_time'])
    # colour ref for Gantt chart
    state_col_ref = {'ED': 'olive', 'Waiting': 'darkgray'}
    count = 0

    cols_whitelist = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'white']
    for key in input_wards.keys():
        state_col_ref[key] = cols_whitelist[count]
        count += 1

    state_col_ref['Discharge'] = 'white'

    sim_end = df_occupancy_log['end_time'].max()

    fig, gnt = plt.subplots()
    gnt.set_xlim(0, sim_end)

    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('simulation time (min)')
    gnt.set_ylabel('patient')

    p_ids = sorted(list(df_occupancy_log['patient_id'].unique()))
    # Setting ticks on y-axis
    gnt.set_yticks([10 * i for i in list(range(1, len(p_ids)))])
    # Labelling tickets of y-axis
    gnt.set_yticklabels(p_ids)

    # Setting graph attribute
    gnt.grid(True)

    # Create the bars
    for ind, entry in df_occupancy_log.iterrows():
        start = entry[2]
        end = entry[3] if entry[3] is not False else sim_end
        duration = end - start
        state = entry[1]
        pos = p_ids.index(entry[0]) * 10
        gnt.broken_barh([(start, duration)], (pos, 10), facecolors=state_col_ref[state])

    list_cols = []
    for key in state_col_ref.keys():
        list_cols.append(mpatches.Patch(color=state_col_ref[key], label=key))

    plt.legend(handles=list_cols, loc='lower right')
    fig.suptitle('Patient journey', fontsize=16)
    plt.show()
    # Save/don't save chart in current dir
    if save_file == "":
        print('Chart not saved')
    else:
        plt.savefig(save_file)
        print('Chart saved as %s in %s' % (save_file, os.getcwd()))
    return df_occupancy_log


def model_run(num_patients, input_wards, tr_matrix, ed_res, save_file=''):
    """
    Old version
    :param num_patients:
    :param input_wards:
    :param tr_matrix:
    :param ed_res:
    :param save_file:
    :return:
    """
    temp_env = simpy.Environment()
    # reset the event log on every run
    global global_event_log
    global_event_log = pd.DataFrame(columns=['patient_id', 'source', 'destination', 'time_stamp', 'event_type'])
    global global_occupancy_log
    global_occupancy_log = []
    # Create a ward system
    ward_system = WardSystem(temp_env, input_wards)
    # Create your ED system
    proc_list = []
    ed = ED(temp_env, num_nurses=ed_res[0], num_rec=ed_res[1])
    # Generate the pathway of each patient
    arrival_time = 0
    for i in range(num_patients):
        """
        # choose a random first ward (for jump around stuff)
        ward_name = random.choice(list(ward_system.wards.keys())[:-1])
        """
        # for sequential everyone starts from the first ward
        ward_name = 'aW_1'
        first_ward = ward_system.wards[ward_name]
        # patient arrival time
        # arrival_time = i
        # random arrival time
        arrival_time += np.random.exponential(30)
        # patient los_dist for each ward (constant for now, needs to be a function within the spell function in the future)
        adm_type = 'NEL' if i % 2 == 0 else 'EL'
        # Generate a process for each patient
        proc_list.append(temp_env.process(Patient(temp_env, i, arrival_time)
                                          .spell(first_ward, ward_system, tr_matrix, ed, adm_type)))
    temp_env.run()
    df_log = gantt_plot(global_occupancy_log, input_wards=input_wards, save_file=save_file)
    return proc_list, df_log, global_event_log


def model_run_until(end_time, arrival_rate, input_wards, tr_matrix, ed_res, save_file='', opt=False):
    """
    Run the model with the following parameters
    :param end_time: duration of the model (in minutes)
    :param arrival_rate: exp. distr. parameter for arrival rate distribution
    :param input_wards: dictionary of wards and their resources e.g. {'Ward_name': N beds}
    :param tr_matrix: Transition matrix square matrix representing the probability of moving from state n to state m
    :param ed_res: resources for emergency department
    :param save_file: optional, enter a name to save a file for the
    :returns: event log, process list and gantt chart plot/data
    """
    temp_env = simpy.Environment()
    # reset the event log on every run
    global global_event_log
    global_event_log = pd.DataFrame(columns=['patient_id', 'source', 'destination', 'time_stamp', 'event_type'])
    global global_occupancy_log
    global_occupancy_log = []
    global T
    T = 0
    # Create a ward system
    ward_system = WardSystem(temp_env, input_wards)
    # Create your ED system
    proc_list = []
    ed = ED(temp_env, num_nurses=ed_res[0], num_rec=ed_res[1])
    # Generate the pathway of each patient
    arrival_time = 0
    temp_id = 0
    while arrival_time < end_time:
        """
        # choose a random first ward (for jump around stuff)
        ward_name = random.choice(list(ward_system.wards.keys())[:-1])
        """
        # for sequential everyone starts from the first ward
        ward_name = 'aW_1'
        first_ward = ward_system.wards[ward_name]
        # random arrival time
        arrival_time += np.random.exponential(arrival_rate)
        # patient los_dist for each ward (constant for now, needs to be a function within the spell function in the future)
        adm_type = 'NEL' if temp_id % 2 == 0 else 'EL'
        # Generate a process for each patient
        proc_list.append(temp_env.process(Patient(temp_env, temp_id, arrival_time)
                                          .spell(first_ward, ward_system, tr_matrix, ed, adm_type)))
        temp_id += 1
    temp_env.run(until=end_time)

    if opt:
        return T
    else:
        df_log = gantt_plot(global_occupancy_log, input_wards=input_wards, save_file=save_file)
        return T, proc_list, df_log, global_event_log


if __name__ == '__main__':
    tr_matrix1 = np.array([[0, 1, 1, 1, 1, 1],  # aW_1
                           [0, 0, 1, 1, 1, 1],  # aW_2
                           [0, 0, 0, 1, 1, 1],  # bW_1
                           [0, 0, 0, 0, 1, 1],  # bW_2
                           [0, 0, 0, 0, 0, 1],  # cW_1
                           [0, 0, 0, 0, 0, 1]])  # dis
    # Dictionary of wards and their bed capacity, 'Discharge' is a dummy ward as there is no resource request for it
    input_wards1 = {'aW_1': 50, 'aW_2': 50, 'bW_1': 20, 'bW_2': 20, 'cW_1': 10, 'Discharge': 1}
    # Run the model
    model_run_until(1440 * 7, 30, input_wards1, tr_matrix1, [1, 1], 'gantt_chart')
