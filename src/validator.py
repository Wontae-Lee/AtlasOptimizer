# External imports
import os
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Internal imports
from .hdf5 import DSMCData


class Validator:

    def __init__(self,
                 path: str,
                 determination_rate: float = 0.1,
                 steady_threshold: float = 0.5,
                 smoothing: int = 5,
                 spline_derivative: int = 1
                 ):
        """
        :param path: The path of the simulation directory.
        (e.g. 'home/user/ATLAS/doc/meetings/000/simulations/cylinder2_0')
        :param determination_rate: The rate of time from the end time to determine the steady state.
        :param steady_threshold: The threshold to determine the steady state.
        :param smoothing: The degree of the spline interpolation.
        :param spline_derivative: The derivative of the spline interpolation.
        """

        self.path = path

        # Get the case name from the path
        while True:
            if path[-1] == '/':
                path = path[:-1]
            else:
                self.case_name = path.split('/')[-1]
                break

        self.determination_rate = determination_rate
        self.steady_threshold = steady_threshold
        self.smoothing = smoothing
        self.spline_derivative = spline_derivative

        # Read the PartAnalyze.csv file
        self.__part_analyze = pd.read_csv(f'{path}/PartAnalyze.csv').fillna(0)
        # Remove rows which contain 0 values
        self.__part_analyze = self.__part_analyze[(self.__part_analyze != 0).all(axis=1)]
        self.__part_analyze_columns = self.__part_analyze.columns
        self.__part_analyze = self.__part_analyze.to_numpy()

        # The columns that can only exist once in PartAnalyze.csv
        self.__time = None
        self.__resolved_timestep = None
        self.__pmax = None
        self.__mean_free_path = None
        self.__max_mcs_over_mfp = None
        self.__resolved_cell_percentage = None
        self.__allocate_columns()

        # The DSMCState files
        self.__dsmc_state_files = [f'{path}/{file}' for file in os.listdir(path) if
                                   'DSMCState' in file and file.endswith('.h5')]
        self.__dsmc_state_files.sort()

    def __allocate_columns(self):
        for index, column in enumerate(self.__part_analyze_columns):
            if 'TIME' in column:
                self.__time = self.__part_analyze[:, index]
            elif 'ResolvedTimestep' in column:
                self.__resolved_timestep = self.__part_analyze[:, index]
            elif 'Pmax' in column:
                self.__pmax = self.__part_analyze[:, index]
            elif 'MeanFreePath' in column:
                self.__mean_free_path = self.__part_analyze[:, index]
            elif 'MaxMCSoverMFP' in column:
                self.__max_mcs_over_mfp = self.__part_analyze[:, index]
            elif 'ResolvedCellPercentage' in column:
                self.__resolved_cell_percentage = self.__part_analyze[:, index]

    def diagnosis(self,
                  steady: bool = True,
                  periodicity: bool = False,
                  collision_probability: bool = True,
                  num_of_particles: bool = True,
                  mcx_over_mfp: bool = True,
                  min_middle_num_of_particles_in_element: int = 40,
                  save: bool = False
                  ):

        self.__diagnosis_steady_state(steady, save, periodicity)

        # Check if the DSMCState files exist
        self.__exist_dsmc_state_files()

        # The following functions need DSMCState files
        if collision_probability:
            self.__diagnosis_collision_probability()
        if num_of_particles:
            self.__diagnosis_num_of_particles(min_middle_num_of_particles_in_element)
        if mcx_over_mfp:
            self.__is_mcx_over_mfp()

    def __diagnosis_steady_state(self, steady: bool, save: bool, periodicity: bool):

        if save:
            os.makedirs(f'./history/{self.case_name}/png', exist_ok=True)
            os.makedirs(f'./history/{self.case_name}/npy', exist_ok=True)

        # Find the steady state time
        steady_state_time_index = int(len(self.__time) * self.determination_rate)
        print(f"""
        
        Validator will check the time in order to determine the steady state.
        if you want to increase the duration that will be checked, 
        
        you can increase the \"determination_rate\" parameter.
        
        from 
            the start time {self.__time[-steady_state_time_index]} 
        to
            the end time {self.__time[-1]}
        """)

        for index, column in enumerate(self.__part_analyze_columns):

            if ('Massflow' in column) or ('Pressure' in column):

                # If the column is a massflow or pressure column, get the values of the column
                # from the steady state time to the end of the simulation
                # x = time, y = column values
                x = self.__time
                y = self.__part_analyze[:, index]

                # Normalize the values
                x_scaled = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()
                y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

                # Spline interpolation
                tck = splrep(x_scaled, y_scaled, s=self.smoothing)

                if save:
                    # Save the normalized y values
                    np.save(f'./history/{self.case_name}/npy/{column}.npy', y_scaled)

                    # Plot the spline interpolation
                    plt.scatter(x_scaled, y_scaled, label=f'{column}')

                    y_spline = list(splev(x_scaled, tck, der=self.spline_derivative))
                    plt.plot(x_scaled, y_spline, label=f'{column} derivative')

                    plt.legend()
                    plt.xlabel(f'Normalized time')
                    plt.ylabel(f'Normalized {column}')
                    plt.title(f'{column} ')
                    plt.savefig(f'./history/{self.case_name}/png/{column}.png')
                    plt.close()

                # Calculate the mean slope of the column values
                # from the steady state time to the end of the simulation
                # slope = dy/dx
                total_slope = 0
                for x_point in x[-steady_state_time_index:]:

                    if periodicity:
                        total_slope += splev(x_point, tck, der=self.spline_derivative)
                    else:
                        total_slope += np.abs(splev(x_point, tck, der=self.spline_derivative))

                mean_slope = total_slope / x.shape[0]

                # If the slope is less than the threshold, the system is in the steady state
                if mean_slope < self.steady_threshold:
                    continue
                elif (mean_slope > self.steady_threshold) and steady:
                    raise Exception(f"""
                    
                    The simulation in the directory {self.path} is not in the steady state.
                    The mean slope of the \"{column}\" column is {mean_slope} which is greater than 
                    the threshold {self.steady_threshold}.
                    
                    Solution:
                    
                    1. you can increase the steady_threshold parameter to roughly determine the steady state.
                    2. you can increase the simulation time to reach the steady state.
                    3. It could happen because you do not have enough particles in the simulation.
                       you can increase the number of particles in the simulation.
                    4. If your simulation is periodic, you can set the periodicity parameter to True.
                       and set the "determination_rate" parameter up to the periodicity time.
                    """)

    def __diagnosis_collision_probability(self):

        # Read the DSMCState files
        dsmc_data = DSMCData(self.__dsmc_state_files[-1])

        for index, column in enumerate(dsmc_data.columns):
            if 'DSMC_MaxCollProb' in column:
                collision_probability = np.max(dsmc_data.data[:, index])
                if collision_probability < 1:
                    continue
                else:
                    raise Exception(f"""
                    
                    The simulation in the directory {self.path} has not a proper collision probability.
                    The maximum collision probability is {collision_probability}.
                    
                    It must be less than 1
            
                    Solution:
                    
                    1. you can decrease the time duration of one time step.
                    
                    """)
            elif 'DSMC_MeanCollProb' in column:
                collision_probability = np.max(dsmc_data.data[:, index]) + np.min(dsmc_data.data[:, index])
                collision_probability /= 2

                if 0.1 < collision_probability < 1:
                    continue
                else:
                    raise Exception(f"""
                    
                    The simulation in the directory {self.path} has not a proper collision probability.
                    The collision probability is {collision_probability}.
                    
                    It must be less than 1
                    it the collision probability is less than 0.2, you can see noise in the output.
                    
                    
                    Solution:
                    
                    1. you can decrease the time duration of one time step.
                    
                    """)

    def __diagnosis_num_of_particles(self, min_middle_num_of_particles_in_element: int):
        # Read the DSMCState files
        dsmc_data = DSMCData(self.__dsmc_state_files[-1])

        for index, column in enumerate(dsmc_data.columns):
            if 'Total_SimPartNum' in column:
                num_of_particles = dsmc_data.data[:, index]

                middle_num_of_particles = (np.max(num_of_particles) + np.min(num_of_particles)) / 2

                if middle_num_of_particles > min_middle_num_of_particles_in_element:
                    continue
                else:
                    raise Exception(f"""
                        
                        The simulation in the directory {self.path} has a low number of particles.
                        The number of particles is {num_of_particles}.
                        
                        It must be more than {middle_num_of_particles}.


                    Solution:
                        
                        1. you can increase the number of particles in the simulation.
                           In order to increase the number of particles, you should decrease "MacroParticleFactor" 
                           in the parameter.ini.
                        
                        """)

    def __is_mcx_over_mfp(self):
        # Read the DSMCState files
        dsmc_data = DSMCData(self.__dsmc_state_files[-1])

        for index, column in enumerate(dsmc_data.columns):
            if 'DSMC_MCS_over_MFP' in column:
                mcx_over_mfp = np.mean(dsmc_data.data[:, index])
                if mcx_over_mfp < 1:
                    continue
                else:
                    raise Exception(f"""
                        
                        The simulation in the directory {self.path} has not a proper 
                        mean collision number over mean free path.
                        
                        The mean collision number over mean free path is {mcx_over_mfp}.
                        
                        It must be less than 1.
                        
                        Solution:
                        
                        1. you can decrease the time duration of one time step.
                        
                        """)

    def __exist_dsmc_state_files(self):
        if len(self.__dsmc_state_files) == 0:
            raise Exception(f"""
                
                There is no DSMCState file in the directory {self.path}.
                
                You have to change the "Part-AnalyzeStep" option in the parameter.ini in order to create DSMCState files.
                """)


if __name__ == '__main__':
    pass
