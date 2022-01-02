import pandas as pd
from tqdm import tqdm

metadata_path = '/home/kajiwara21/work/sed/meta/test_meta_strong.csv'
meta_df = pd.read_csv(metadata_path)
filenames = meta_df['filename'].unique().tolist()

results = {
    'air_conditioner': 0,
    'car_horn': 0,
    'children_playing': 0,
    'dog_bark': 0,
    'drilling': 0,
    'engine_idling': 0,
    'gun_shot': 0,
    'jackhammer': 0,
    'siren': 0,
    'street_music': 0
}

total_time = {
    'air_conditioner': 0,
    'car_horn': 0,
    'children_playing': 0,
    'dog_bark': 0,
    'drilling': 0,
    'engine_idling': 0,
    'gun_shot': 0,
    'jackhammer': 0,
    'siren': 0,
    'street_music': 0
}

for filename in tqdm(filenames):
    annotation_df = meta_df[meta_df['filename'] == filename]
    for i, row1 in annotation_df.iterrows():
        onset1 = row1['onset']
        offset1 = row1['offset']

        for j, row2 in annotation_df.iterrows():
            if i == j:
                continue

            onset2 = row2['onset']
            offset2 = row2['offset']
            if offset1 >= onset2 and offset2 >= onset1:
                start = onset1
                if onset1 < onset2:
                    start = onset2

                end = offset1
                if offset1 > offset2:
                    end = offset2

                results[row1['event_label']] += (end - start)
                total_time[row1['event_label']] += (offset1 - onset1)

for k, v in results.items():
    print(k, v/total_time[k])
