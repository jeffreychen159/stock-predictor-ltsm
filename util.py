import datetime

# Changes a str of date to datetime
def str_to_datetime(s): 
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    
    return datetime.datetime(year=year, month=month, day=day)

# Changes a string of date and time to datetime
def time_to_datetime(s): 
    split = s.split(' ')
    date = split[0]
    time = split[1]
    
    split_date = date.split('-')
    year, month, day = int(split_date[0]), int(split_date[1]), int(split_date[2])
    
    split_hour = time.split(':')
    hour, minute = int(split_hour[0]), int(split_hour[1])
    
    return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)