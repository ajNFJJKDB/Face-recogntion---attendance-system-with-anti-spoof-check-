import pandas as pd
from datetime import datetime, timedelta
import os
import random

# Define students and monitoring times
student_ids = ['student1', 'student2', 'student3', 'student4', 'student5', 'student6']
monitoring_intervals = [
    ('08:00', '9:00'),
    ('10:25', '11:25'),
    ('12:25', '13:25')
]

# Define the date for simulation
simulation_date = '2024-05-14'

# Probability that a student is present during a break
presence_probability = 0.7

# Number of entries per student per interval
entries_per_student = 100

# Ensure at least 2 students are present in each interval
min_students_per_interval = 2

# Function to ensure directory existence
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to generate random timestamps within the interval
def generate_random_timestamps(start_time_str, end_time_str, date_str, num_entries):
    start_time = datetime.strptime(f"{date_str} {start_time_str}", '%Y-%m-%d %H:%M')
    end_time = datetime.strptime(f"{date_str} {end_time_str}", '%Y-%m-%d %H:%M')
    interval_seconds = (end_time - start_time).total_seconds() / num_entries
    return [start_time + timedelta(seconds=i * interval_seconds + random.uniform(0, interval_seconds)) for i in range(num_entries)]

# Generate simulated IN and OUT timestamps
def generate_simulated_timestamps(student_ids, monitoring_intervals, date_str, presence_probability, entries_per_student, min_students_per_interval):
    in_records = []
    out_records = []
    
    for interval_start, interval_end in monitoring_intervals:
        students_present = random.sample(student_ids, k=max(min_students_per_interval, int(len(student_ids) * presence_probability)))
        for student_id in students_present:
            in_timestamps = generate_random_timestamps(interval_start, interval_end, date_str, entries_per_student)
            out_timestamps = [ts + timedelta(seconds=random.uniform(1, 120)) for ts in in_timestamps]  # OUT timestamps 1 to 120 seconds after IN
            
            for in_ts, out_ts in zip(in_timestamps, out_timestamps):
                in_records.append({'student_id': student_id, 'timestamp': in_ts, 'direction': 'IN'})
                out_records.append({'student_id': student_id, 'timestamp': out_ts, 'direction': 'OUT'})
    
    return pd.DataFrame(in_records), pd.DataFrame(out_records)

# Generate the simulated data
simulated_in_timestamps_df, simulated_out_timestamps_df = generate_simulated_timestamps(
    student_ids, monitoring_intervals, simulation_date, presence_probability, entries_per_student, min_students_per_interval)

# Save the simulated files
output_dir = 'C:/Users/balaj/OneDrive/Desktop/work/ml_project'
ensure_dir(output_dir)
simulated_in_timestamps_df.to_csv(os.path.join(output_dir, 'simulated_in_timestamps.csv'), index=False)
simulated_out_timestamps_df.to_csv(os.path.join(output_dir, 'simulated_out_timestamps.csv'), index=False)

print(f'Simulated IN and OUT timestamps have been saved to {output_dir}')
