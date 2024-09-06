import pandas as pd
from datetime import datetime
import os

# Function to ensure directory existence
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_weekday(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return date.strftime('%A')

def find_period(timestamp, periods):
    time_str = timestamp.strftime('%H:%M')
    for period, (start, end) in periods.items():
        if start <= time_str < end:
            return period
    return None

def assign_subject(row, weekday, weekly_schedule, periods):
    if row['Period'] is not None:
        day_periods = weekly_schedule[weekday]
        period_index = list(periods.keys()).index(row['Period'])
        return day_periods[period_index] if period_index < len(day_periods) else None
    return None

def export_attendance_for_date(specified_date):
    # Convert specified date to weekday
    weekday = get_weekday(specified_date)

    # Define base directory for that specific date
    base_directory = f'C:/Users/balaj/OneDrive/Desktop/work/ml_project/timestamp/{specified_date}'
    if not os.path.exists(base_directory):
        print(f"No data available for {specified_date}")
        return

    # Load the timestamp data from the specified date directory
    in_file = os.path.join(base_directory, 'in_timestamps.csv')
    out_file = os.path.join(base_directory, 'out_timestamps.csv')
    if not os.path.exists(in_file) or not os.path.exists(out_file):
        print(f"Incomplete data for {specified_date}")
        return

    in_timestamps_df = pd.read_csv(in_file, parse_dates=['timestamp'])
    out_timestamps_df = pd.read_csv(out_file, parse_dates=['timestamp'])

    # Define school hours and periods
    periods = {
        'First': ('8:00', '9:00'),
        'Second': ('09:00', '10:00'),
        'Third': ('10:25', '11:25'),
        'Fourth': ('11:25', '12:25'),
        'Fifth': ('13:30', '14:30'),
        'Sixth': ('14:30', '15:30')
    }

    # Define subjects for each weekday
    weekly_schedule = {
        'Monday': ['subject 1', 'subject 2', 'subject 3', 'subject 4', 'subject 5', 'subject 6'],
        'Tuesday': ['subject 2', 'subject 3', 'subject 4', 'subject 5', 'subject 6', 'subject 1'],
        'Wednesday': ['subject 3', 'subject 4', 'subject 5', 'subject 6', 'subject 1', 'subject 2'],
        'Thursday': ['subject 4', 'subject 5', 'subject 6', 'subject 1', 'subject 2', 'subject 3'],
        'Friday': ['subject 5', 'subject 6', 'subject 1', 'subject 2', 'subject 3', 'subject 4']
    }

    # Process timestamp files
    in_timestamps_df['Period'] = in_timestamps_df['timestamp'].apply(lambda ts: find_period(ts, periods))
    out_timestamps_df['Period'] = out_timestamps_df['timestamp'].apply(lambda ts: find_period(ts, periods))

    # Combine IN and OUT timestamps
    attendance_df = pd.merge(in_timestamps_df, out_timestamps_df, on=['student_id', 'Period'], suffixes=('_in', '_out'))

    # Determine if present
    attendance_df['Present'] = attendance_df['timestamp_out'] > attendance_df['timestamp_in']

    # Assign subjects to periods based on the day
    attendance_df['Subject'] = attendance_df.apply(lambda row: assign_subject(row, weekday, weekly_schedule, periods), axis=1)

    # Export the attendance for that specific day
    daily_summary = attendance_df.groupby(['student_id', 'Subject'])['Present'].any().reset_index()
    daily_summary.to_csv(os.path.join(base_directory, f'{weekday}_attendance.csv'), index=False)
    print(f'Attendance exported for {specified_date} {weekday} in {weekday}_attendance.csv')

# User specifies the date here in 'YYYY-MM-DD' format
specified_date = '2024-05-22'
export_attendance_for_date(specified_date)
