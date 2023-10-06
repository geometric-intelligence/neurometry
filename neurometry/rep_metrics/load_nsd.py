from benchmarks import NSDBenchmark
import pandas as pd


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

    neural_data = {}
    for subject_id in subject_ids:
        subject_neural_data = {}
        subject_dataframe = voxel_metadata[voxel_metadata["subj_id"] == subject_id]
        for region in rois:
            region_voxel_idxs = subject_dataframe[
                subject_dataframe[region] == True
            ].index
            subject_neural_data[region] = response_data.loc[region_voxel_idxs]
            print(
                f"Subject {subject_id} has {len(subject_neural_data[region])} voxels in region {region}"
            )
        neural_data[subject_id] = subject_neural_data

    return neural_data
