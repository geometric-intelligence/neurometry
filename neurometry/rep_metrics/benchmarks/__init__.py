import os, sys, shutil
import numpy as np
import pandas as pd
from PIL import Image

def fisherz(r, eps=1e-5):
    return np.arctanh(r-eps)

def fisherz_inv(z):
    return np.tanh(z)

def average_rdms(rdms):
    return 1 - fisherz_inv(fisherz(np.stack([1 - rdm for rdm in rdms])).mean(axis = 0, keepdims = True).squeeze())

class NSDBenchmark():
    def __init__(self, image_set = 'shared1000', voxel_set = 'OTC-only',
                 train_test_split = False, clean_rdms_only = True,
                 anatomical_roi_subset = None, functional_roi_subset = None):
        
        print('Now loading the {} image set and the {} voxel set...'.format(image_set, voxel_set))
        
        self.name = '_'.join([image_set, voxel_set])
        self.index_name = 'voxel_id'
        
        path_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        
        response_path = f'{path_dir}/voxel_sets/{self.name}/voxel_betas.csv'
        metadata_path = f'{path_dir}/voxel_sets/{self.name}/voxel_metadata.csv'
        stimulus_path = f'{path_dir}/image_sets/{image_set}.csv'
        self.image_root = os.path.join(path_dir, 'image_sets', image_set)
        
        path_set = [response_path, metadata_path, stimulus_path]
        if not all([os.path.exists(path) for path in path_set]):
            raise ValueError('Paths poorly specified. Please check code.')
            
        self.stimulus_data = pd.read_csv(stimulus_path)
        self.n_stimuli = len(self.stimulus_data)
        
        self.response_data = pd.read_csv(response_path).set_index(self.index_name)
        self.metadata = pd.read_csv(metadata_path).set_index(self.index_name)
        
        self.stimulus_data['image_path'] = self.image_root + '/' + self.stimulus_data.image_name
        
        anatomical_rois = ['early','ventral','lateral','EVC','OTC']
        functional_rois = ['V1v','V1d','V2v','V2d','V3v','V3d','hV4',
                           'FFA-1','FFA-2','OFA','EBA','FBA-1','FBA-2',
                                    'OPA','PPA', 'VWFA-1','VWFA-2','OWFA']
        
        self.anatomical_rois = anatomical_rois if not anatomical_roi_subset else functional_roi_subset
        self.functional_rois = functional_rois if not functional_roi_subset else anatomical_roi_subset
        
        self.all_rois = [roi for roi in self.metadata.columns if roi in 
                         self.anatomical_rois + self.functional_rois]
        
        self.rdm_indices = self.get_rdm_indices()
        self.rdms = self.get_rdms()

        if clean_rdms_only:
            clean_rdms = {}
            for roi in self.rdms:
                clean_rdms[roi] = {}
                for subj_id in self.rdms[roi]:
                    if np.sum(np.isnan(self.rdms[roi][subj_id])) == 0:
                        clean_rdms[roi][subj_id] = self.rdms[roi][subj_id]

            self.rdms = clean_rdms

        if train_test_split == True:
            self.response_data = {'train': self.response_data.iloc[:,::2], 
                                  'test': self.response_data.iloc[:,1::2]}

            self.stimulus_data = {'train': self.stimulus_data.iloc[::2,:],
                                  'test': self.stimulus_data.iloc[1::2,:]}

            self.rdms = self.get_splithalf_rdms()

        
    def get_sample_stimulus(self, image_index = None):
        if image_index is None:
            image_index = np.random.randint(self.n_stimuli)

        sample_image_path = self.stimulus_data.image_name[image_index]

        return Image.open(os.path.join(self.image_root, sample_image_path))

    get_stimulus = get_sample_stimulus

    def get_rdm_indices(self, roi_subset = None, row_number = False):
        metadata = self.metadata
        if self.index_name in metadata.columns:
            metadata = metadata.set_index(self.index_name)

        if not roi_subset:
            roi_subset = self.all_rois
            
        if row_number:
            metadata = metadata.reset_index()

        rdm_indices = {}
        for roi in roi_subset:
            roi_subset = metadata[metadata[roi] == 1]
            rdm_indices[roi] = {}
            for subj_id in roi_subset.subj_id.unique():
                subj_id_subset = roi_subset[roi_subset['subj_id'] == subj_id]
                rdm_indices[roi][subj_id] = subj_id_subset.index.to_numpy()

        return rdm_indices

    def get_rdms(self, roi_subset = None, include_group_average = False):
        responses = self.response_data
        if self.index_name in responses.columns:
            responses = responses.set_index(self.index_name)

        if not roi_subset:
            roi_subset = self.all_rois

        if not self.rdm_indices:
            self.rdm_indices = self.get_rdm_indices(roi_subset = roi_subset)

        brain_rdms = {}
        for roi in self.rdm_indices:
            brain_rdms[roi] = {}
            for subj_id in self.rdm_indices[roi]:
                target_responses = responses.loc[self.rdm_indices[roi][subj_id]]
                if target_responses.shape[0] > 10:
                    brain_rdms[roi][subj_id] = 1 - np.corrcoef(target_responses.transpose())

        if include_group_average:
            for roi in brain_rdms:
                brain_rdms[roi]['group_average'] = average_rdms(brain_rdms[roi].values())

        return brain_rdms

    def get_splithalf_rdms(self):
        split_rdms = {}
        for roi in self.rdms:
            split_rdms[roi] = {}
            for subj_id in self.rdms[roi]:
                split_rdms[roi][subj_id] = {'train': self.rdms[roi][subj_id][::2,::2],
                                            'test': self.rdms[roi][subj_id][1::2,1::2]}

        return split_rdms