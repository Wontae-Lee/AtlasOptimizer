import h5py
import numpy as np


class DSMCData:

    def __init__(self, h5_path: str):
        with h5py.File(f'{h5_path}', 'r') as dsmc_file:
            self.data = dsmc_file['ElemData']
            self.data = np.array(self.data)
            self.columns = ['Total_Velo:0', 'Total_Velo:1', 'Total_Velo:2', 'Total_TempTransX',
                            'Total_TempTransY', 'Total_TempTransZ', 'Total_NumberDensity',
                            'Total_TempVib', 'Total_TempRot', 'Total_TempElec', 'Total_SimPartNum',
                            'Total_TempTransMean', 'DSMC_MaxCollProb', 'DSMC_MeanCollProb',
                            'DSMC_MCS_over_MFP']


class StateData:

    def __init__(self, h5_path):
        with h5py.File(f'{h5_path}', 'r') as state_file:
            # Print the keys at the root level of the file

            self.particle_data = state_file['PartData']
            self.particle_data = np.array(self.particle_data)

            self.dg_solution = state_file['DG_Solution']
            self.dg_solution = np.array(self.dg_solution)

            self.element_data = state_file['ElemData']
            self.element_data = np.array(self.element_data)

            self.element_time = state_file['ElemTime']
            self.element_time = np.array(self.element_time)

            self.particle_int = state_file['PartInt']
            self.particle_int = np.array(self.particle_int)


class MeshData:

    def __init__(self, h5_path: str):
        self.columns = ['x', 'y', 'z']
        with h5py.File(h5_path, 'r') as f:
            self.mesh = f['ElemBarycenters']
            self.mesh = np.array(self.mesh)