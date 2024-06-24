import os
import numpy as np
import pandas as pd


class Tool:

    def __init__(self,
                 path: str,
                 adaptive_type: str,
                 specie: int,
                 surfaceflux: int,
                 pressure: float,
                 cell_size: float,
                 particles_in_element: float = 1.0,
                 boltzmann_constant: float = 1.380649e-23,
                 temperature: float = 300
                 ):

        self.path = path
        self.__find_parameter_ini()
        self.__set_adaptive_type(adaptive_type)
        self.__specie = specie
        self.__surfaceflux = surfaceflux
        self.__pressure = pressure
        self.__cell_size = cell_size
        self.__particles_in_element = particles_in_element
        self.__boltzmann_constant = boltzmann_constant
        self.__temperature = temperature

        self.__macro_particles = f"Part-Species{specie}-MacroParticleFactor"
        self.__check_none(self.__macro_particles)

        self.__set_option()

        self.__target = None
        self.__compute_macro_particles(pressure)

        self.__desired_value = None
        self.__fitted = False

    def __set_adaptive_type(self, adaptive_type: str):
        if adaptive_type not in ["pressure"]:
            raise ValueError(f"""
            
            The adaptive type {adaptive_type} is not valid.
            The adaptive type must be "pressure" currently.
            
            """)
        if adaptive_type == "pressure":
            self.__adaptive_type = "Pressure"

    def __compute_macro_particles(self, pressure: float):

        # Compute the macro particle factor
        macro_particle_factor = (pressure / (self.__boltzmann_constant * self.__temperature)) \
                                * self.__cell_size ** 3 / self.__particles_in_element

        self.__change_value(self.__macro_particles, int(macro_particle_factor))

    def __set_option(self):
        self.__option = f"Part-Species{self.__specie}-Surfaceflux{self.__surfaceflux}-Adaptive-{self.__adaptive_type}"
        self.__check_none(self.__option)
        self.__change_value(self.__option, self.__pressure)

    def set_target(self, output_type: str, value: float = None, characteristic_length: float = None,
                   analytical_time: float = None):

        if output_type == "flux":
            self.__target = f"{self.__adaptive_type}-Spec-00{self.__specie}-SF-00{self.__surfaceflux}"
            self.__desired_value = value

        elif output_type == "Kn":
            self.__target = "MeanFreePath"
            self.__desired_value = value * characteristic_length * analytical_time

    def __find_parameter_ini(self):
        """Find the parameter.ini file in the path"""
        for file in os.listdir(self.path):
            if file.endswith(".ini"):
                if file == "parameter.ini":
                    self.__parameter_ini = os.path.join(self.path, file)
                    return True

        raise FileNotFoundError(f"""
            
            parameter.ini file not found in the path
            {self.path}
            
            """)

    def __check_none(self, option: str):
        # Read the parameter.ini file
        with open(self.__parameter_ini, "r") as file:
            lines = file.readlines()

        # Update the parameter.ini file
        for line in lines:
            if (f"{option}" in line) and (f"{option}=None" not in line):
                raise ValueError(f"""
                    
                    The option {option} is already defined in the parameter.ini file.
                    
                    """)

    def __change_value(self, option: str, value: float):
        """Change the value of the option in the parameter.ini file"""
        # Read the parameter.ini file
        with open(self.__parameter_ini, "r") as file:
            lines = file.readlines()

        # Update the parameter.ini file
        for index, line in enumerate(lines):
            if f"{option}" in line:
                lines[index] = f"{option}={value}\n"

        # Write the parameter.ini file
        with open(self.__parameter_ini, "w") as file:
            file.writelines(lines)

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

    def __read_part_analyze(self):
        # Read the PartAnalyze.csv file
        self.__part_analyze = pd.read_csv(f'{self.path}/PartAnalyze.csv').fillna(0)
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

    def fit(self, rate: float = 0.5, accuracy: float = 0.05):
        self.__read_part_analyze()

        for index, column in enumerate(self.__part_analyze_columns):
            if self.__target in column:
                steady_value = self.__part_analyze[-1, index]

                error_rate = (steady_value - self.__desired_value) / self.__desired_value

                if np.abs(error_rate) < accuracy:
                    self.__fitted = True
                    return
                else:
                    self.__pressure = self.__pressure + self.__pressure * rate * error_rate
                    self.__pressure = np.abs(self.__pressure)
                    self.__change_value(self.__option, self.__pressure)
                    self.__compute_macro_particles(self.__pressure)

    def is_fitted(self):
        return self.__fitted
