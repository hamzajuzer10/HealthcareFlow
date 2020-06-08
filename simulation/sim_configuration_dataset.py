import datetime

def generate_sim_config():

    config_dict = {'actual_vs_forecast': 'forecast',
                   'hospital_init': True,
                   'warmup_period_mins': 3600,
                   'sim_initial_timestamp': datetime.datetime(2019, 10, 4, 9, 0, 0),
                   'sim_end_timestamp': datetime.datetime(2019, 10, 10, 17, 0, 0)
    }

    verify_timestamp(config_dict['sim_initial_timestamp'], config_dict['sim_end_timestamp'])


    return config_dict

def verify_timestamp(initial_timestamp, end_timestamp):
    """check end timestamp is bigger than initial timestamp"""

    if initial_timestamp >= end_timestamp:
        return ValueError

    return 0

