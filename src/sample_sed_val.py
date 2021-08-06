import sed_eval
import dcase_util
import pandas as pd
from tqdm import tqdm

# reference_event_list = dcase_util.containers.MetaDataContainer(
#     [
#         {
#             'event_label': 'car_horn',
#             'onset': 4.2457340000000015,
#             'offset': 4.333698000000001,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'car_horn',
#             'onset': 8.947329,
#             'offset': 10.0,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'air_conditioner',
#             'onset': 4.4248460000000005,
#             'offset': 6.8157630000000005,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'air_conditioner',
#             'onset': 6.775143000000001,
#             'offset': 10.0,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'jackhammer',
#             'onset': 5.8694770000000025,
#             'offset': 7.8146010000000015,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'engine_idling',
#             'onset': 6.5990020000000005,
#             'offset': 10.0,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'dog_bark',
#             'onset': 9.183627,
#             'offset': 10.0,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'street_music',
#             'onset': 4.295691609977324,
#             'offset': 9.328616780045351,
#             'filename': 'soundscape_validate_bimodal111.wav',
#         },
#     ]
# )

# estimated_event_list = dcase_util.containers.MetaDataContainer(
#     [
#         {
#             'event_label': 'air_conditioner',
#             'onset': 7.9005895691609975,
#             'offset': 9.264761904761905,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'air_conditioner',
#             'onset': 9.380861678004536,
#             'offset': 9.978775510204082,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'dog_bark',
#             'onset': 8.463673469387755,
#             'offset': 10.0,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'jackhammer',
#             'onset': 6.0836281179138325,
#             'offset': 7.732244897959184,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'street_music',
#             'onset': 4.295691609977324,
#             'offset': 9.328616780045351,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#         {
#             'event_label': 'street_music',
#             'onset': 9.427301587301587,
#             'offset': 9.537596371882087,
#             'filename': 'soundscape_validate_bimodal443.wav',
#         },
#     ]
# )

ref_df = pd.read_csv('../meta/valid_meta_strong.csv')
est_df = pd.read_csv('../pred.csv')

reference_event_list = dcase_util.containers.MetaDataContainer(ref_df.T.to_dict().values())
estimated_event_list = dcase_util.containers.MetaDataContainer(est_df.T.to_dict().values())

segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
    event_label_list=reference_event_list.unique_event_labels,
    time_resolution=0.001
)
event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
    event_label_list=reference_event_list.unique_event_labels,
    t_collar=0.250
)

for filename in tqdm(reference_event_list.unique_files):
    reference_event_list_for_current_file = reference_event_list.filter(
        filename=filename
    )

    estimated_event_list_for_current_file = estimated_event_list.filter(
        filename=filename
    )

    segment_based_metrics.evaluate(
        reference_event_list=reference_event_list_for_current_file,
        estimated_event_list=estimated_event_list_for_current_file
    )

    event_based_metrics.evaluate(
        reference_event_list=reference_event_list_for_current_file,
        estimated_event_list=estimated_event_list_for_current_file
    )

print(segment_based_metrics.results()['overall'])
print(event_based_metrics.results()['overall'])
