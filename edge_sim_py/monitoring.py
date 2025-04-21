from edge_sim_py import *
import os
import networkx as nx
import msgpack
import itertools
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import requests
####################################################

"""
The line
datasets[file.replace(".msgpack", "")] = pd.DataFrame(msgpack.unpackb(data_file.read(), strict_map_key=False))
does not create new files on disk.
Instead, it reads the content of existing MessagePack files, converts them into DataFrames, and
stores these DataFrames in a dictionary in memory.
"""
# Adjust display settings to show all columns and increase width
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# Set the maximum number of rows to display
pd.set_option('display.max_rows', 100000)  # Set this to the desired number, or None for unlimited rows

##############################################################################################################################
def load_msgpack_logs(logs_directory):
    """
    Load msgpack log files from the specified directory into pandas DataFrames.

    Args:
        logs_directory (str): Path to the directory containing msgpack log files.

    Returns:
        dict: Dictionary of pandas DataFrames, where the keys are the log file names.
    """
    datasets = {}
    dataset_files = [file for file in os.listdir(logs_directory) if ".msgpack" in file]

    for file in dataset_files:
        try:
            with open(os.path.join(logs_directory, file), 'rb') as data_file:
                datasets[file.replace(".msgpack", "")] = pd.DataFrame(
                    msgpack.unpackb(data_file.read(), strict_map_key=False))
        except (msgpack.UnpackException, ValueError) as e:
            print(f"Error loading file {file}: {e}")

    return datasets

##################################
### Determining name algorithm ###
##################################

# scheduling_algorithm = "lapse"
scheduling_algorithm = "EDF"
logs_directory = f"{os.getcwd()}/logs/algorithm={scheduling_algorithm};dataset=dataset1;"
datasets = load_msgpack_logs(logs_directory)
##############################################################################################################################


"""
EdgeServer detailed logs
"""
# Access the last row of the DataFrame
last_row = datasets["EdgeServer"].iloc[-1]

# Retrieve the value of the 'Time Step' attribute
time_step_value = last_row['Time Step']

# Find the largest number in the 'Instance ID' column
max_instance_id = datasets["EdgeServer"]["Instance ID"].max()

# ## showing all the columns of the "EdgeServer"
# print(datasets["EdgeServer"].columns)
selected_columns = ['Object', 'Time Step', 'Instance ID', 'Coordinates', 'Available', 'RAM', 'Disk', 'Processor Power Demand', 'RAM Demand', 'Disk Demand', 'Services', 'Registries', 'Layers', 'Images', 'Download Queue', 'Waiting Queue', 'Power Consumption']
# print(datasets["EdgeServer"][selected_columns])

### Filter rows with specific condition - last "Time Step"
filtered_df = datasets["EdgeServer"][(datasets["EdgeServer"]["Time Step"] == time_step_value)]
filtered_selected_df = filtered_df[selected_columns]
print(filtered_selected_df)
print()


# Sum of the 'Power Consumption' of all edge servers and number of services in edge computing
total_power_consumption = round((filtered_df[selected_columns]['Power Consumption'].sum()), 2)
total_unique_services = datasets["Service"]["Instance ID"].nunique()
print()



"""
Services detailed logs
"""
# Access the last row of the DataFrame
last_row = datasets["Service"].iloc[-1]

# Retrieve the value of the 'Time Step' attribute
time_step_value = last_row['Time Step']

# Find the largest number in the 'Instance ID' column
max_instance_id = datasets["Service"]["Instance ID"].max()
# showing all the columns of the "Service"
# print(datasets["Service"].columns)
# selecting specific columns
selected_columns_service = ['Object', 'Time Step', 'Instance ID', 'Available', 'Application', 'Server', 'Last Migration']
### filtering the rows
filtered_col_service = datasets["Service"][(datasets["Service"]["Time Step"] == (time_step_value))]
print(filtered_col_service[selected_columns_service])

# Initialize a counter for 'miss'
service_miss_count = 0
# Iterate over each row in the 'Deadline Status' column
for status in filtered_col_service[selected_columns_service]['Available']:
    # Check if 'miss' is present in the dictionary values
    if status == False:
        service_miss_count += 1


"""
Edge Users detailed logs
"""
# Access the last row of the DataFrame
last_row = datasets["User"].iloc[-1]

# Retrieve the value of the 'Time Step' attribute
time_step_value = last_row['Time Step']

# Find the largest number in the 'Instance ID' column
max_instance_id = datasets["User"]["Instance ID"].max()

# Initialize a counter for users who met the deadline and total users
meet_deadline_count = 0
total_users = 0

# showing all the columns of the "User"
# print(datasets["User"].columns)

# selecting specific columns
selected_columns_user = ['Object', 'Time Step', 'Instance ID', 'Required Deadline', 'Round Trip Time', 'In Server Processing Time', 'Response Time', 'Deadline Status']
# # printing the selected columns from User
## print(datasets["User"][selected_columns_user])
# # # filtering the rows
filtered_col_user = datasets["User"][(datasets["User"]["Time Step"] == time_step_value)]
# print(filtered_col_user[selected_columns_user])

# # Initialize a counter for 'miss'
might_miss_count = 0
### Iterate over each row in the 'Deadline Status' column
for status in filtered_col_user[selected_columns_user]['Deadline Status']:
    total_users += 1
    # Check if 'miss' is present in the dictionary values
    if not status.values():
        might_miss_count += 1

print(f"\n{service_miss_count} out of {(len(filtered_col_service[selected_columns_service]['Available']))} services are experienced failures (missed/lost/failed), potentially affecting {might_miss_count} of the {total_users} users.")