from benchmarks import NSDBenchmark
import pandas as pd
from collections import defaultdict


def load_nsd(target_regions):
    combined_response_data = []
    combined_metadata = []

    for region in target_regions:
        benchmark = NSDBenchmark(*f"shared1000_{region}-only".split("_"))

        response_data = benchmark.response_data.copy()
        voxel_metadata = benchmark.metadata.copy()

        combined_response_data.append(response_data)
        combined_metadata.append(voxel_metadata)

    response_data = pd.concat(combined_response_data).drop_duplicates()
    voxel_metadata = pd.concat(combined_metadata).drop_duplicates()
    stimulus_data = benchmark.stimulus_data

    functional_rois = benchmark.functional_rois

    return response_data, voxel_metadata, stimulus_data, functional_rois


def get_neural_data(subjects, rois, voxel_metadata, response_data):
    subject_ids = [int(s.split("subj")[1]) for s in subjects]
    neural_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for i, subject_id in enumerate(subject_ids):
        subject_rois = rois[subjects[i]]
        subject_dataframe = voxel_metadata[voxel_metadata["subj_id"] == subject_id]
        for region in subject_rois:
            region_voxel_idxs = subject_dataframe[
                subject_dataframe[region] == True
            ].index
            neural_data_subject_roi = response_data.loc[region_voxel_idxs]
            print(
                f"Subject {subject_id} has {len(neural_data_subject_roi)} voxels in region {region}"
            )
            x_values = neural_data_subject_roi.index.to_series().str.split("-", expand=True)[1].astype(int)
            
            left_hemisphere = neural_data_subject_roi[x_values < 40]
            right_hemisphere = neural_data_subject_roi[x_values >= 40]
            neural_data[subject_id]["left"][region] = left_hemisphere
            neural_data[subject_id]["right"][region] = right_hemisphere

    return neural_data

