import os
import pandas as pd


class Tool:

    def __init__(self, path: str, adaptive_type: str, specie: int, surfaceflux: int, value: float):
        self.path = path
        self.__define_option(adaptive_type, specie, surfaceflux)
        self.__define_target(adaptive_type, specie, surfaceflux)
        self.__desired_value = value
        self.__fitted_value = value
        self.__parameter_ini = None
        self.__fitted = False
        self.__find_parameter_ini()
        self.__check_option_none()
        self.__change_value(self.__option, self.__desired_value)

    def __define_option(self, adaptive_type: str, specie: int, surfaceflux: int):

        if adaptive_type not in ["pressure", "massflow"]:
            raise ValueError(f"""
            
            The adaptive type {adaptive_type} is not valid.
            The adaptive type must be either 'pressure' or 'massflow'.
            
            """)

        self.__option = f"Part-Species{specie}-Surfaceflux{surfaceflux}-Adaptive-{adaptive_type}"

    def __define_target(self, adaptive_type: str, specie: int, surfaceflux: int):
        if adaptive_type not in ["pressure", "massflow"]:
            raise ValueError(f"""
                    
                    The adaptive type {adaptive_type} is not valid.
                    The adaptive type must be either 'pressure' or 'massflow'.
                    
                    """)
        self.__target = f"{adaptive_type}-Spec-00{specie}-SF-00{surfaceflux}"

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

    def __check_option_none(self):

        # Read the parameter.ini file
        with open(self.__parameter_ini, "r") as file:
            lines = file.readlines()

        # Update the parameter.ini file
        for line in lines:
            if (f"{self.__option}" in line) and (f"{self.__option}=None" not in line):
                raise ValueError(f"""
                
                The option {self.__option} is already defined in the parameter.ini file.
                
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

    def fitting(self, rate: float = 0.5, accuracy: float = 0.05):
        self.__read_part_analyze()

        for index, column in enumerate(self.__part_analyze_columns):
            if self.__target in column:
                steady_value = self.__part_analyze[-1, index]

                error = abs(steady_value - self.__desired_value) / self.__desired_value

                if error < accuracy:
                    self.__fitted = True
                    return
                else:
                    self.__fitted_value = self.__fitted_value + rate * (self.__desired_value - steady_value)
                    self.__change_value(self.__option, self.__fitted_value)

    def is_fitted(self):
        return self.__fitted
