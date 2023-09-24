import os, sys, shutil
import numpy as np
import pandas as pd
from PIL import Image

### NSDBenchmark -------------------------------------------------

class NSDBenchmark():
    def __init__(self, image_set='NSD_shared1000', voxel_set=['EVC','OTC'],
                 anatomical_roi_subset=None, functional_roi_subset=None, **kwargs):
        
        voxel_set_name = '-'.join(voxel_set) if isinstance(voxel_set, list) else voxel_set
        print(f"Now loading the {image_set} image set and the {voxel_set_name} voxel set...")
        
        self.name = f'{image_set}_{voxel_set_name}'
        self.image_set = image_set
        self.voxel_set = voxel_set
        self.index_name = 'voxel_id'
        
        path_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if kwargs.pop('root', None) is not None:
            path_dir = kwargs.pop('root')
            
        stimulus_path = f'{path_dir}/stimulus/{image_set}.csv'    
        
        path_set = [stimulus_path]
        if isinstance(voxel_set, str) and voxel_set in ['EVC','OTC']:
            response_path = f'{path_dir}/response/{self.name}/voxel_betas.csv'
            metadata_path = f'{path_dir}/response/{self.name}/voxel_metas.csv'
            path_set += [response_path, metadata_path]
        
        if isinstance(voxel_set, list):
            response_path = {}
            metadata_path = {}
            for vset in voxel_set:
                response_path[vset] = f'{path_dir}/response/{image_set}_{vset}/voxel_betas.csv'
                metadata_path[vset] = f'{path_dir}/response/{image_set}_{vset}/voxel_metas.csv'
                path_set += [response_path[vset], metadata_path[vset]]
    
        self.image_root = os.path.join(path_dir, 'stimulus', image_set)
        
        if not all([os.path.exists(path) for path in path_set]):
            print(path_set)
            raise ValueError('Paths poorly specified. Please check code.')
            
        self.file_paths = {'response_data': response_path,
                           'metadata': metadata_path,
                           'stimulus_data': stimulus_path}
        
        if kwargs.pop('absolute_paths', False):
            self.file_paths = get_absolute_paths(self.file_paths)
        
        self.response_data = self.load_data('response_data')
        self.metadata = self.load_data('metadata')
        self.stimulus_data = pd.read_csv(stimulus_path)
        self.n_stimuli = len(self.stimulus_data)
        
        self.stimulus_data['image_path'] = self.image_root + '/' + self.stimulus_data.image_name
        
        self.image_paths = self.stimulus_data.image_path.to_list()
        self.image_descs = self.stimulus_data.coco_captions.to_list()
        
        anatomical_rois = ['early','ventral','lateral','EVC','OTC']
        functional_rois = ['V1v','V1d','V2v','V2d','V3v','V3d','hV4',
                           'FFA-1','FFA-2','OFA','EBA','FBA-1','FBA-2',
                                    'OPA','PPA', 'VWFA-1','VWFA-2','OWFA']
        
        self.anatomical_rois = anatomical_rois if not anatomical_roi_subset else functional_roi_subset
        self.functional_rois = functional_rois if not functional_roi_subset else anatomical_roi_subset
        
        self.all_rois = [roi for roi in self.metadata.columns if roi in 
                         self.anatomical_rois + self.functional_rois]
        
        roi_voxel_counts = {roi: (self.metadata[roi] == True).sum() for roi in self.all_rois}
        self.roi_voxel_counts = dict(sorted(roi_voxel_counts.items(), key=lambda x: x[1], reverse=True))
        
        self.rdm_indices = self.get_rdm_indices()
        
    def __repr__(self):
        roi_info = '\n Top ROI Constituents:'
        for i, (roi, count) in enumerate(self.roi_voxel_counts.items()):
            if i+1 > 3: break
            roi_info += f'\n   {roi}: {count} Voxels'
        
        return (f'Natural Scenes Dataset Sample (Subj01)' +
                f'\n Macro ROI(s): {self.voxel_set}' +
                f'\n # Probe Stimuli: {self.n_stimuli}' +
                f'\n # Responding Voxels: {len(self.response_data)}' + roi_info)
              
        
    def load_data(self, which_data):
        data_path = self.file_paths[which_data]
        if isinstance(data_path, str):
            return pd.read_csv(data_path).set_index(self.index_name)
        
        if isinstance(data_path, dict):
            response_data = []
            for key, value in data_path.items():
                response_data += [pd.read_csv(data_path[key])
                                  .set_index(self.index_name)]
                
            return pd.concat(response_data)

    def get_file_paths(self):
        return self.file_paths

    def get_stimulus(self, index=None):
        if index is None:
            index = np.random.randint(self.n_stimuli)

        sample_image_path = self.stimulus_data.image_name[index]

        return Image.open(os.path.join(self.image_root, sample_image_path))

    get_sample_stimulus = get_stimulus

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

### Helpers + Convenience -------------------------------------------------
    
def get_absolute_paths(paths_dict):
    for key, value in paths_dict.items():
        if isinstance(value, dict):
            paths_dict[key] = get_absolute_paths(value)
        else:
            paths_dict[key] = os.path.abspath(value)
    return paths_dict