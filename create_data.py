# create_data.py
"""
Creates a small synthetic deliveries dataset for the slot-prediction MVP.
Run: python create_data.py
This writes deliveries.csv in the same folder.
"""

import pandas as pd
import random

def random_timeslot():
    return random.choice(['08-10', '10-12', '12-14', '14-17', '17-20'])

def random_dow():
    return random.choice(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

def gen_row(i):
    customer_id = random.randint(1,200)
    hour_pref = random.choice([0,1])  # 1 if customer gave preferred hour, 0 otherwise
    timeslot = random_timeslot()
    dow = random_dow()
    work = random.choice([0,1])       # 1 if customer likely at work (less available)
    is_weekend = 1 if dow in ['Sat','Sun'] else 0
    attempts_before = random.choice([0,1,2])
    # Simple rule to decide "success" (1 = successful delivery in that slot)
    # This is synthetic: you will replace with real data later.
    success_chance = 0.6
    # make weekends and people not at work more likely to succeed
    if is_weekend == 1 and work == 0:
        success_chance = 0.9
    elif work == 1:
        success_chance = 0.4
    # slight penalty if attempts_before > 0
    if attempts_before > 0:
        success_chance -= 0.15 * attempts_before
    success = 1 if random.random() < success_chance else 0

    return {
        'customer_id': customer_id,
        'dow': dow,
        'work': work,
        'attempts_before': attempts_before,
        'hour_pref_provided': hour_pref,
        'timeslot': timeslot,
        'success': success
    }

def create_dataset(n_rows=1000, out_file='deliveries.csv'):
    rows = [gen_row(i) for i in range(n_rows)]
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file} with {len(df)} rows")
    print("Sample rows:")
    print(df.sample(5).to_string(index=False))

if __name__ == "__main__":
    create_dataset(n_rows=1000)
