import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from scipy.interpolate import CubicSpline
import itertools
import Surface_confined_inference as sci
from pathlib import Path
from kneed import KneeLocator
import tabulate
import re
from string import ascii_uppercase
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.widgets import Button, Slider

class ExperimentEvaluation:
    """
    A class for evaluating and analyzing electrochemical experiments with support for
    multiple file types, experiment configurations, and visualization methods.
    """
    
    def __init__(self, data_locations, input_params, boundaries, common=None, **kwargs):
        """
        Initialize the ExperimentEvaluation class.
        
        Parameters:
        -----------
        parameter_dict : dict
            Dictionary containing parameter lists for different experiment types.
            Expected keys: 'SWV', 'FTACV', 'optimisation'.
        data_locations : str or list
            Path(s) to the directory(ies) containing experimental data files.
        input_params : dict
            Dictionary containing experiment parameters.
        extras : dict, optional
            Extra parameters to apply to all experiments (e.g., temperature, area).
            Default is None.
        """
        if "SWV_e0_shift" not in kwargs:
            kwargs["SWV_e0_shift"]=False
        self.options=kwargs
        self.input_parameters = input_params
        self.classes = {}
        self.all_keys = []
        self.data_locations = [data_locations] if isinstance(data_locations, str) else data_locations
        self.boundaries=boundaries
        self.grouping_keys=None
        self.common = common or {
            "Temp": 278,
            "area": 0.036,
            "N_elec": 1,
            "Surface_coverage": 1e-10
        }
        
        # Initialize classes for all experiments
        for location in self.data_locations:
            files = os.listdir(location)
            
            # Process each experiment type (e.g., FTACV, SWV)
            for experiment in self.input_parameters.keys():
                # Use recursive helper function to traverse the nested dictionaries
                self._process_experiment_conditions(
                    experiment=experiment, 
                    conditions_dict=self.input_parameters[experiment],
                    files=files,
                    location=location,
                    current_key_parts=[experiment],
                    current_path=[]
                )   
        self.class_keys=list(self.classes.keys())
        self.all_parameters=set()
        self.all_harmonics=set()
        for key in self.class_keys:
            self.all_parameters=self.all_parameters.union(self.classes[key]["class"].optim_list)
            if self.classes[key]["class"].experiment_type in ["FTACV","PSV"]:
                self.all_harmonics=self.all_harmonics.union(set(self.classes[key]["class"].Fourier_harmonics))
            else:
                if self.options["SWV_e0_shift"]==True:
                    if "SquareWave" in self.classes[key]["class"].experiment_type:
                        if "anodic" not in key and "cathodic" not in key:
                            raise ValueError("If SWV_e0_shift is set to True, then all SWV experiments must be identified as anodic or cathodic, not {0}".format(key))
        self.all_harmonics=list(self.all_harmonics)

        self.all_parameters=list(self.all_parameters)
    def _process_experiment_conditions(self, experiment, conditions_dict, files, location, 
                                 current_key_parts, current_path):
        """
        Recursively process nested experiment conditions to initialize experiment classes.
        
        Parameters:
        -----------
        experiment : str
            The experiment type (e.g., 'FTACV', 'SWV').
        conditions_dict : dict
            Dictionary containing conditions at the current level.
        files : list
            List of files in the data location.
        location : str
            Path to the data location.
        current_key_parts : list
            Parts of the experiment key accumulated so far.
        current_path : list
            Path through the conditions dict to reach the current point.
        """
        # Check if we've reached a leaf node (actual experiment parameters)
        keys=conditions_dict.keys()
        missing=set(["Parameters", "Options", "Zero_params"])-set(keys)
        if len(missing)>0 and len(missing)<3:
            raise KeyError("Missing the following elements from the experimental dictionary: {0}".format(" / ".join(list(missing))))
        if "Parameters" in keys and "Options" in keys and "Zero_params" in keys:
            experiment_params=conditions_dict["Parameters"]
            # We've reached experiment parameters, create the experiment class
            experiment_key = "-".join(current_key_parts)
            self.all_keys.append(experiment_key)
            print(f"Initializing: {experiment_key}")

            
            # Add extra parameters
            for key in self.common:
                experiment_params[key] = self.common[key]
            
            # Set experiment type
            init_exp = "FTACV" if experiment == "FTACV" else "SquareWave"
            
            # Create experiment class
            self.classes[experiment_key] = {
                "class": sci.RunSingleExperimentMCMC(
                    init_exp,
                    experiment_params,
                    problem="inverse",
                    normalise_parameters=True
                )
            }
            try:
                self.classes[experiment_key]["class"].boundaries = self.boundaries
                if init_exp=="SquareWave":
                    self.classes[experiment_key]["class"].fixed_parameters = {"Cdl": 0}
                for key in conditions_dict["Options"].keys():
                    self.classes[experiment_key]["class"].__setattr__(key, conditions_dict["Options"][key])
                
                # Set boundaries
                
            except Exception as e:
                raise KeyError("Error processing {0}:{1}".format(experiment_key, str(e)))

            
            # Find matching data file using all conditions in the key
            key_parts = experiment_key.split("-")
            matching_files = [
                x for x in files if all(part in x for part in key_parts)
            ]
            
            if not matching_files:
                print("Warning: No matching file found for {0}".format(experiment_key))
                return
            
            file = matching_files[0]
            
            # Process data based on experiment type
  
            self._process_data(experiment, experiment_key, location, file, conditions_dict)
        else:
            # We haven't reached a leaf node, continue recursion
            for cond, next_level in conditions_dict.items():
                new_key_parts = current_key_parts + [cond]
                new_path = current_path + [cond]
                
                if isinstance(next_level, dict):
                    self._process_experiment_conditions(
                        experiment=experiment,
                        conditions_dict=next_level,
                        files=files,
                        location=location,
                        current_key_parts=new_key_parts,
                        current_path=new_path
                    )
                
                    
    
    def _process_data(self, experiment, experiment_key, location, file, conditions_dict):
        """
        Process data for a specific experiment.
        
        Parameters:
        -----------
        experiment : str
            Experiment type ('FTACV' or 'SWV').
        experiment_key : str
            Unique experiment identifier.
        location : str
            Directory containing the data file.
        file : str
            Filename of the experimental data.
        """
    
        try:
            data = np.loadtxt(os.path.join(location, file), delimiter=",")
        except:
            data = np.loadtxt(os.path.join(location, file))
            
        if experiment == "FTACV":
            self._process_ftacv_data(experiment_key, data, conditions_dict["Zero_params"])
        else:  # SWV
            self._process_swv_data(experiment_key, data, conditions_dict["Zero_params"])
      
    
    def _process_ftacv_data(self, experiment_key, data, zero_params):
        """
        Process FTACV data.
        
        Parameters:
        -----------
        experiment_key : str
            Unique experiment identifier.
        data : numpy.ndarray
            Experimental data array.
        """        
        # Normalize data
        cls=self.classes[experiment_key]["class"]
        time=data[:,0]
        current=data[:,1]
        norm_current = cls.nondim_i(current)
        norm_time = cls.nondim_t(time)
        
        # Store in class dictionary
        self.classes[experiment_key]["data"] = norm_current
        self.classes[experiment_key]["times"] = norm_time
        self.classes[experiment_key]["FT"] = cls.experiment_top_hat(norm_time, norm_current)
        
        # Generate zero point for error calculation
        dummy_zero_class = sci.RunSingleExperimentMCMC(
            "FTACV",
            cls._internal_memory["input_parameters"],
            problem="forwards",
            normalise_parameters=False
        )
        dummy_zero_class.dispersion_bins = [1]
        dummy_zero_class.optim_list = cls.optim_list
        
        worst_case = dummy_zero_class.simulate(
            zero_params, 
            norm_time
        )
        ft_worst_case = cls.experiment_top_hat(norm_time, worst_case)
        
        self.classes[experiment_key]["zero_point"] = sci._utils.RMSE(worst_case, norm_current)
        self.classes[experiment_key]["zero_sim"]=worst_case
        self.classes[experiment_key]["zero_point_ft"] = sci._utils.RMSE(ft_worst_case, self.classes[experiment_key]["FT"])

    def _process_swv_data(self, experiment_key, data, zero_params):
        """
        Process SWV data.
        
        Parameters:
        -----------
        experiment_key : str
            Unique experiment identifier.
        data : numpy.ndarray
            Experimental data array.
        """
        current = data[:-1, 1]
        cls = self.classes[experiment_key]["class"]
        
        # Configure class
        
        
        
        
        # Calculate times and voltages
        times = cls.calculate_times()
        voltage = cls.get_voltage(times)
        pot = np.array([voltage[int(x)] for x in cls._internal_memory["SW_params"]["b_idx"]])
        
        # Apply baseline correction
        if zero_params is not None:
            signal_region = zero_params["potential_window"]
            before = np.where((pot < signal_region[0]))
            after = np.where((pot > signal_region[1]))
            
            noise_data = []
            noise_spacing = zero_params["thinning"]
            roll = zero_params["smoothing"]
            midded_current = sci._utils.moving_avg(current, roll)
            
            for sequence in [pot, midded_current]:
                catted_sequence = np.concatenate([
                    sequence[before][roll+10-1::noise_spacing],
                    sequence[after][roll+10-1::noise_spacing]
                ])
                noise_data.append(catted_sequence)
            
            sort_args = np.argsort(noise_data[0])
            sorted_x = [noise_data[0][x] for x in sort_args]
            sorted_y = [noise_data[1][x] for x in sort_args]
            
            # Apply cubic spline for baseline correction
            CS = CubicSpline(sorted_x, sorted_y)
            
            # Store normalized data
            self.classes[experiment_key]["data"] = cls.nondim_i(current - CS(pot))
        else:
             self.classes[experiment_key]["data"] = cls.nondim_i(current)
        self.classes[experiment_key]["times"] = times
        self.classes[experiment_key]["zero_sim"]=np.zeros(len(current))
        self.classes[experiment_key]["zero_point"] = sci._utils.RMSE(np.zeros(len(current)), self.classes[experiment_key]["data"])
    
    def initialise_grouping(self, group_list):
        self.group_list=group_list
        numeric_qualifiers=["lesser", "geq", "between", "equals"]
        self.experiment_grouping={}
        for i in range(0, len(group_list)):
            experiment_list=[x.split("-") for x in list(self.class_keys) if group_list[i]["experiment"] in x]
            naughty_list=[]
            empty_key=["{0}:{1}".format(x, group_list[i][x]) for x in ["experiment", "type"]]
            if "numeric" in group_list[i]:
                
                for key in group_list[i]["numeric"].keys():
                    found_unit=False
                    in_key="_{0}-".format(key)
                    for expkey in self.class_keys:
                        if in_key in expkey or expkey[-len(in_key)+1:]==in_key[:-1]:
                            found_unit=True
                    if found_unit==False:
                        raise ValueError("{0} not found in any experiments".format(key))

                        
                    if "-" in key:
                        raise ValueError("'-'is not allowed in groupkeys ({0})".format(key))
                    qualifiers=list(group_list[i]["numeric"][key].keys())
                    if len(qualifiers)>1:
                        raise ValueError("{0} in {1}, {2} has more than one qualifier".format(key, empty_key[0], empty_key[1]))
                    else:
                        qualifier=qualifiers[0]
                        
                    if qualifier not in numeric_qualifiers:
                        raise ValueError("{0} not in allowed qualifiers - lesser, geq, between".format(qualifiers[0]))
                    if qualifier!="between":
                        qualifier_value=float(group_list[i]["numeric"][key][qualifier])
                        empty_key+=["%s:%d%s" % (qualifier, qualifier_value, key)] 
                    else:
                        qualifier_value=[float(x) for x in group_list[i]["numeric"][key][qualifier]]
                        empty_key+=["%s:%d~%d%s" % (qualifier, qualifier_value[0],qualifier_value[1], key)]
                    for j in range(0, len(experiment_list)):
                        current_exp=experiment_list[j]
                        get_numeric=float([x for x in current_exp if key in x][0].split("_")[0])
                        if qualifier=="lesser":
                            if get_numeric>=qualifier_value:
                               naughty_list.append(j)
                        elif qualifier =="geq":
                            if get_numeric<qualifier_value:
                               naughty_list.append(j)
                        elif qualifier=="between":
                            if get_numeric>qualifier_value[1] or get_numeric<qualifier_value[0]:
                               naughty_list.append(j)
                        elif qualifier=="equals":
                            if get_numeric!=qualifier_value:
                               naughty_list.append(j)
            if "match" in group_list[i]:
                
                for match in group_list[i]["match"]:
                    empty_key+=["match:{0}".format(match)] 
                    for j in range(0, len(experiment_list)):
                        if match not in experiment_list[j]:
                           naughty_list.append(j)
            final_key="-".join(empty_key)
            
            final_experiment_list=[experiment_list[i] for i in range(0, len(experiment_list)) if i not in naughty_list]
            self.experiment_grouping[final_key]=["-".join(x) for x in final_experiment_list]
        self.grouping_keys=list(self.experiment_grouping.keys())     
        self.group_dict=dict(zip(self.grouping_keys, self.group_list)) 
    def get_zero_point_scores(self):
        zero_dict={}
        for classkey in self.class_keys:
            zero_dict[classkey]=self.classes[classkey]["zero_sim"]
        return self.simple_score(zero_dict)
    def initialise_simulation_parameters(self, grouped_param_dictionary=None):
        if self.grouping_keys==None:
                raise ValueError("Please call initialise_grouping() first")
        if grouped_param_dictionary==None:
           
            self.parameter_map={x:copy.deepcopy(self.all_parameters) for x in self.grouping_keys}
            return self.all_parameters
        new_all_parameters=[]
        self.parameter_map={x:copy.deepcopy(self.all_parameters) for x in self.grouping_keys}
        for key in grouped_param_dictionary:
            if key not in self.all_parameters:
                raise ValueError("{0} not in optim_list of any class".format(key))
            all_idx=list(itertools.chain(*grouped_param_dictionary[key]))
            set_idx=set(all_idx)#existing_values
            required_idx=list(range(0, len(self.grouping_keys)))#required_values
            if len(set_idx)<len(required_idx):
                missing_values=list(set(required_idx).difference(set_idx))
                raise ValueError("{0} in parameter grouping assignment missing indexes for {1}".format(key, " ".join(["{0} (index {1})".format(self.grouping_keys[x],x) for x in missing_values])))

            if len(all_idx)!=len(required_idx):
                diffsum=sum(np.diff(list(set_idx)))
                print(diffsum, required_idx[-1])
                if (diffsum-1)!=required_idx[-1]:
                    raise ValueError("{0} (in {1}) in parameter grouping assignment contains duplicates".format(all_idx, key))
                else:
                    raise ValueError("{0} (in {1}) in parameter grouping assignment contains more indexes that then number of groups ({2})".format(all_idx, key, len(required_idx)))


            new_all_parameters+=["{0}_{1}".format(key, x+1) for x in range(0, len(grouped_param_dictionary[key]))]
            for m in range(0,len(grouped_param_dictionary[key])):
                element=grouped_param_dictionary[key][m]
                for j in range(0, len(element)):
                    group_key=self.grouping_keys[element[j]]
                    p_idx=self.parameter_map[group_key].index(key)
                    self.parameter_map[group_key][p_idx]="{0}_{1}".format(key, m+1)
        common_params=[x for x in self.all_parameters if x not in grouped_param_dictionary]
        
        self.all_parameters=new_all_parameters+common_params
        if self.options["SWV_e0_shift"]==True:
            if "E0_mean" not in grouped_param_dictionary and "E0" not in grouped_param_dictionary:
                if "E0_mean" in self.all_parameters:
                    self.all_parameters+=["E0_mean_offset"]
                elif "E0" in self.all_parameters:
                    self.all_parameters+=["E0_offset"]
            else:
                if "E0_mean" in grouped_param_dictionary:
                    target="E0_mean"
                elif "E0" in grouped_param_dictionary:
                    target="E0"
                for groupkey in self.grouping_keys:
                    exp=[self.classes[x]["class"].experiment_type=="SquareWave" for x in self.experiment_grouping[groupkey]]
                    if all(exp)==True:
                        optim_list=self.parameter_map[group_key]
                        param=[x for x in optim_list if re.search(target+r"_\d", x)][0]+"_offset"
                        if param not in self.all_parameters:
                            self.all_parameters+=[param]
                    elif any(exp)==True:
                        raise ValueError("If SWV_e0_shift is set to True, all members of a SWV group have to be SquareWave experiments")
                    
    def parse_input(self, parameters):
        in_optimisation=False
        try:
            values=copy.deepcopy([parameters.get(x) for x in self.all_parameters])
            valuedict=dict(zip(self.all_parameters, values))
            in_optimisation=True
        except:
            valuedict=dict(zip(self.all_parameters, copy.deepcopy(parameters)))
        optimisation_parameters={}
        for group_key in self.grouping_keys:
            if hasattr(self, "parameter_map"):
                parameter_list=self.parameter_map[group_key]
            else:
                raise ValueError("Need to run `initialise_simulation_parameters()` function first")
            sim_values={}
            for classkey in self.experiment_grouping[group_key]:
                cls=self.classes[classkey]["class"]
                for param in parameter_list:
                    if param in cls.optim_list:
                        sim_values[param]=valuedict[param]
                    elif "_offset" in param:
                        continue
                    else:
                        found_parameter=False
                        for param2 in cls.optim_list:
                            changed_param=param2+"_"
                            if changed_param in param:
                                sim_values[param2]=valuedict[param]
                                found_parameter=True
                                break
                                
                        
                for param in self.all_parameters:
                    if self.classes[classkey]["class"].experiment_type!="SquareWave":
                        continue
                    elif self.options["SWV_e0_shift"]==True:
                        if "offset" in param:
                            idx=param.find("_offset")
                            true_param=param[:idx]
                            if true_param not in cls.optim_list:
                                for param2 in cls.optim_list:
                                    changed_param=param2+"_"
                                    if changed_param in param:
                                        true_param=param2
                            if "anodic" in classkey:
                                sim_values[true_param]+=valuedict[param]
                            elif "cathodic" in classkey:
                                sim_values[true_param]-=valuedict[param]
                            else:
                                raise ValueError("If SWV_e0_shift is set to True, then all SWV experiments must be identified as anodic or cathodic, not {0}".format(key))
                optimisation_parameters[classkey]=[sim_values[x] for x in cls.optim_list]
        for key in self.class_keys:
            if key not in optimisation_parameters:
                raise KeyError("{0} not added to optimisation list, check that at least one group includes it".format(key))
        return optimisation_parameters   
    def evaluate(self, parameters):
        simulation_params_dict=self.parse_input(parameters)
        simulation_values_dict={}
        for classkey in self.class_keys:
            cls=self.classes[classkey]["class"]
            
            sim_params=simulation_params_dict[classkey]
            simulation_values_dict[classkey]=cls.simulate(sim_params, self.classes[classkey]["times"])
        return simulation_values_dict
    def optimise_simple_score(self, parameters):
        simulation_values=self.evaluate(parameters)
        return self.simple_score(simulation_values)
    def simple_score(self, simulation_values_dict):
        score_dict={}
        
        for groupkey in self.grouping_keys:
            current_score=0
            for classkey in self.experiment_grouping[groupkey]:
                cls=self.classes[classkey]["class"]
                if "type:ft" not in groupkey:
                    classscore=sci._utils.RMSE(simulation_values_dict[classkey], self.classes[classkey]["data"])
                else:
                    classscore=sci._utils.RMSE(cls.experiment_top_hat(self.classes[classkey]["times"],simulation_values_dict[classkey]), self.classes[classkey]["FT"])
                if "scaling" in self.group_dict[groupkey]:
                    classscore=self.scale(classscore, groupkey, classkey)
                current_score+=classscore
            score_dict[groupkey]=current_score
        return score_dict
                   
    def scale(self, value, groupkey, classkey):
        value=copy.deepcopy(value)
        cls=self.classes[classkey]["class"]
        if "divide" in self.group_dict[groupkey]["scaling"]:
            for param in self.group_dict[groupkey]["scaling"]["divide"]:
                value/=cls._internal_memory["input_parameters"][param]
        if "multiply" in self.group_dict[groupkey]["scaling"]:
            for param in self.group_dict[groupkey]["scaling"]["multiply"]:
                value*=cls._internal_memory["input_parameters"][param]  
        return value
        
    def check_grouping(self,):
        self.plot_results([], savename=None, show_legend=True)
    def add_legend(self, ax, groupkey, target_cols=3):
        num_labels=len(self.experiment_grouping[groupkey])
        if num_labels<target_cols:
            num_cols=num_labels
            height=1
        else:
            num_cols=target_cols
            if num_labels%target_cols!=0:
                extra=1
            else:
                extra=0
            height=int(num_labels//target_cols)+extra
        ylim=ax.get_ylim()
        
        ax.set_ylim([ylim[0]-(abs(max(ylim)))*0.1*height, ylim[1]])
        ax.legend(loc="lower center", bbox_to_anchor=[0.5, 0], ncols=num_cols, handlelength=1)
                

    def plot_stacked_time(self, axis, data_list, **kwargs):
        if "colour" not in kwargs:
            kwargs["colour"]=None
        if "linestyle" not in kwargs:
            kwargs["linestyle"]="-"
        if "alpha" not in kwargs:
            kwargs["alpha"]=1
        if "label_list" not in kwargs:
            kwargs["label_list"]=None
        if "patheffects" not in kwargs:
            kwargs["patheffects"]=None
        if "lw" not in kwargs:
            kwargs["lw"]=None
        current_len=0
        line_list=[]
        for i in range(0, len(data_list)):
            xaxis=range(current_len, current_len+len(data_list[i]))
            if kwargs["label_list"] is not None:
                label=kwargs["label_list"][i]
            else:
                label=None
            l1, =axis.plot(xaxis, data_list[i], label=label, alpha=kwargs["alpha"], linestyle=kwargs["linestyle"], color=kwargs["colour"], lw=kwargs["lw"], path_effects=kwargs["patheffects"])
            current_len+=len(data_list[i])
            line_list.append(l1)
        axis.set_xticks([])
        
        return axis, line_list
    def process_harmonics(self, harmonics_list, **kwargs):
        """
        Process harmonics data and calculate all appropriate scalings.
        
        Parameters:
        -----------
        harmonics_list : list
            List of harmonics data to process
        **kwargs : dict
            Additional keyword arguments:
            - harmonics: list of harmonics to process (default: None)
            - scale: whether to scale harmonics (default: True)
            - residual: whether to compute residuals (default: False)
            
        Returns:
        --------
        dict
            Dictionary containing processed data and parameters for plotting
        """
        # Set default values if not provided
        if "harmonics" not in kwargs:
            kwargs["harmonics"] = self.all_harmonics
        if "scale" not in kwargs:
            kwargs["scale"] = True
        if "residual" not in kwargs:
            kwargs["residual"] = False
        if "additional_maximum" not in kwargs:
            kwargs["additional_maximum"]=0
            
        # Process the harmonics list
        arrayed = np.array(harmonics_list)
        maximum = max(np.max(arrayed, axis=None), kwargs["additional_maximum"])
        #print(maximum)
        #print(np.max(arrayed, axis=None),kwargs["additional_maximum"], "*")
        # Get dimensions
        num_plots = arrayed.shape[0]
        num_experiments = arrayed.shape[1]
        num_harmonics = len(kwargs["harmonics"]) if kwargs["harmonics"] is not None else arrayed.shape[2]
        
        # Validate residual option
        if kwargs["residual"] == True and num_plots != 2:
            raise ValueError("Can only do two sets of harmonics for a residual plot")
        
        # Calculate scaling factors and offsets
        scaled_data = []
        for m in range(num_experiments):
            exp_data = []
            for i in range(num_harmonics):
                harmonic_data = []
                # Calculate maximum for current harmonic across all plots
                current_maximum = np.max(np.array([arrayed[x][m][i,:] for x in range(num_plots)]), axis=None)
                # Calculate offset for stacking
                offset = (num_harmonics - i) * 1.1 * maximum
                # Calculate scaling ratio
                ratio = maximum / current_maximum if kwargs["scale"] else 1
                
                for j in range(num_plots):
                    xdata = range(len(arrayed[j][m][i,:]))
                    ydata = ratio * arrayed[j][m][i,:] + offset
                    harmonic_data.append((xdata, ydata))
                
                exp_data.append(harmonic_data)
            scaled_data.append(exp_data)
        
        # Prepare result dictionary
        result = {
            "scaled_data": scaled_data,
            "dimensions": {
                "num_plots": num_plots,
                "num_experiments": num_experiments,
                "num_harmonics": num_harmonics
            },
            "maximum": maximum
        }
        #scaled data is returned as experiment -> harmonic -> plot
        return result

    def plot_scaled_harmonics(self, axis, processed_data, **kwargs):
        """
        Plot scaled harmonics data.
        
        Parameters:
        -----------
        axis : matplotlib.axes.Axes
            Axis to plot on
        processed_data : dict
            Dictionary containing processed data from process_harmonics
        **kwargs : dict
            Additional keyword arguments for styling:
            - colour: list of colors for each plot
            - linestyle: list of line styles
            - lw: list of line widths
            - patheffects: list of path effects
            - alpha: list of alpha values
            - label_list: list of labels for each experiment
            - utils_colours: list of colors to use for different experiments
            
        Returns:
        --------
        tuple
            (axis, line_list) - the axis and list of plotted lines
        """
        scaled_data = processed_data["scaled_data"]
        dimensions = processed_data["dimensions"]
        num_plots = dimensions["num_plots"]
        num_experiments = dimensions["num_experiments"]
        num_harmonics = dimensions["num_harmonics"]
        
        # Process styling parameters
        # Set default values
        if "colour" not in kwargs:
            kwargs["colour"] = [None]
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = ["-"]
        if "lw" not in kwargs:
            kwargs["lw"] = [None]
        if "patheffects" not in kwargs:
            kwargs["patheffects"] = [None]
        if "alpha" not in kwargs:
            kwargs["alpha"] = [1]
        if "label_list" not in kwargs:
            kwargs["label_list"] = None
            
        # Validate styling parameters
        style_keys = ["colour", "linestyle", "alpha", "lw", "patheffects"]
        for key in style_keys:
            if isinstance(kwargs[key], list) is False:
                raise ValueError(f"{key} needs to be wrapped into a list")
            
            if len(kwargs[key]) != num_plots:
                if len(kwargs[key]) == 1:
                    kwargs[key] = [kwargs[key][0] for _ in range(num_plots)]
                else:
                    raise ValueError(f"{key} needs to be the same length as the number of plots")
        
        # Prepare for plotting
        current_len = 0
        line_list = []
        
        # Get default colors
        utils_colours = kwargs.get("utils_colours", None)
        
        # Plot each experiment
        for m in range(num_experiments):
            
            
            # Plot each harmonic
            for i in range(num_harmonics):
                current_line_list = []
                # Plot each dataset
                for j in range(num_plots):
                    xdata, ydata = scaled_data[m][i][j]
                    
                    # Adjust x-axis for continuous plotting
                    xaxis = [x + current_len for x in xdata]
                    
                    # Set label for the first line of each experiment
                    if i == 0 and j == 0 and kwargs["label_list"] is not None:
                        label = kwargs["label_list"][m]
                    else:
                        label = None
                    
                    # Set color
                    if isinstance(kwargs["colour"][j], list) or isinstance(kwargs["colour"][j], np.ndarray):
                        colour = kwargs["colour"][j]
                    elif kwargs["colour"][j] is None:
                        # Use utils_colours if provided, otherwise use default
                        colour = utils_colours[m] if utils_colours is not None else f"C{m}"
                    else:
                        colour = kwargs["colour"][j]
                    
                    # Plot the line
                    l1, = axis.plot(
                        xaxis, 
                        ydata, 
                        label=label, 
                        alpha=kwargs["alpha"][j], 
                        linestyle=kwargs["linestyle"][j], 
                        color=colour, 
                        lw=kwargs["lw"][j], 
                        path_effects=kwargs["patheffects"][j]
                    )
                    
                    current_line_list.append(l1)
                line_list.append(current_line_list)
            # Update current length for the next harmonics
            if len(scaled_data[m][i][0][0]) > 0:
                current_len += len(scaled_data[m][i][0][0])
            
               
        
        # Remove x-axis ticks
        axis.set_xticks([])
        
        return axis, line_list
    def results(self, input_list, **kwargs):
        if "pre_saved" not in kwargs:
            kwargs["pre_saved"]=False     
        if kwargs["pre_saved"]==False:
            if isinstance(input_list[0], list) is False:
                input_list=[input_list]
        else:
            if isinstance(input_list, dict) is True:
                input_list=[input_list]
        self.plot_results(self.process_simulation_dict(input_list, pre_saved=kwargs["pre_saved"]), **kwargs)  
    def process_simulation_dict(self, input_list, pre_saved):
        total_all_simulations=[]
        for p in input_list:
            if pre_saved==False:
                simulation_vals=self.evaluate(p)
            else:
                simulation_vals=p
            groupsims={}
            for i in range(0, len(self.grouping_keys)):
                groupkey=self.grouping_keys[i]
                groupsims[groupkey]=[self.scale(simulation_vals[x], groupkey, x) for x in self.experiment_grouping[groupkey]]
        total_all_simulations.append(groupsims)
        return total_all_simulations
    def plot_results(self, simulation_values, **kwargs):
        linestyles=[ "dashed", "dashdot","dotted",]
        if "target_key" not in kwargs:
            target_key=[None]
        else:
            if isinstance(kwargs["target_key"], str):   
                target_key=[kwargs["target_key"]]
            elif isinstance(kwargs["target_key"], list):
                target_key=kwargs["target_key"]
        if "simulation_labels" not in kwargs:
            kwargs["simulation_labels"]=None
        elif len(kwargs["simulation_labels"]) != len(simulation_values):
            raise ValueError("simulation_labels ({0}) not same length as provided simulation_values ({1}) ".format(len(kwargs["simulation_labels"]), len(simulation_values)))
        if "savename" not in kwargs:
            kwargs["savename"]=None
        if "show_legend" not in kwargs:
            kwargs["show_legend"]=False
        if "axes" not in kwargs:
            if len(self.grouping_keys)%2!=0:
                num_cols=(len(self.grouping_keys)+1)/2
                fig,axes=plt.subplots(2, int(num_cols))
                axes[1, -1].set_axis_off()
            else:
                num_cols=len(self.grouping_keys)/2
                fig,axes=plt.subplots(2, int(num_cols))
        else:
            axes=kwargs["axes"]
        self.plot_line_dict={}
        defaults={
                "cmap":mpl.colormaps['plasma'],
                "foreground":"black",
                "lw":0.4,
                "strokewidth":3,
                "colour_range":[0.2, 0.75]
                
            }
        if "sim_plot_options" not in kwargs:
            kwargs["sim_plot_options"]="simple"
        elif kwargs["sim_plot_options"]=="default":
            pass
        else:
            for key in defaults.keys():
                if key not in kwargs["sim_plot_options"]:
                    kwargs["sim_plot_options"]=defaults[key]
        if kwargs["sim_plot_options"]!="simple":
            plot_colours=kwargs["sim_plot_options"]["cmap"](
                np.linspace(kwargs["sim_plot_options"]["colour_range"][0],
                kwargs["sim_plot_options"]["colour_range"][1],
                len(simulation_values))
            )
            path_effects=[pe.Stroke(linewidth=kwargs["sim_plot_options"]["strokewidth"], foreground=kwargs["sim_plot_options"]["foreground"]), pe.Normal()]
        else:
            path_effects=None      
        if "interactive_mode" not in kwargs:
            kwargs["interactive_mode"]=False
        if kwargs["interactive_mode"]==True:
            if isinstance(simulation_values[0], dict) is False or len(simulation_values)>1:
                raise ValueError("In interactive mode, you can only submit a single dictioanry of simulation values")
            self.simulation_plots={"maxima":{}, "data_harmonics":{}}
        for i in range(0, len(self.grouping_keys)):
                
                groupkey=self.grouping_keys[i]
                ax=axes[i%2, i//2]
                
                if groupkey in target_key:
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(4)
                        ax.tick_params(width=4)
                ax.set_title(groupkey, fontsize=8)
                all_data=[self.scale(self.classes[x]["data"], groupkey, x) for x in self.experiment_grouping[groupkey]]
                all_times=[self.classes[x]["times"] for x in self.experiment_grouping[groupkey]]
                label_list=[",".join(x.split("-")[1:]) for x in self.experiment_grouping[groupkey]]
                all_simulations=[x[groupkey] for x in simulation_values]
                
                    
                
                if "type:ft" not in groupkey:
                    self.plot_stacked_time(ax, all_data, label_list=label_list)
                    for q in range(0, len(all_simulations)):
                        if kwargs["sim_plot_options"]=="simple":
                            axis, time_lines=self.plot_stacked_time(ax, all_simulations[q], alpha=0.75, linestyle=linestyles[q%4], colour="black")
                        else:
                            axis, time_lines=self.plot_stacked_time(ax, all_simulations[q], alpha=0.75, linestyle=linestyles[q%4], colour=plot_colours[q], patheffects=path_effects, lw=kwargs["sim_plot_options"]["lw"])
                        if kwargs["interactive_mode"]==True:
                            self.simulation_plots[groupkey]=time_lines
                else:
                    num_experiments = len(all_data)
                    num_simulations = len(all_simulations) + 1  # +1 for the data
                    
                    # Initialize harmonics_list with proper dimensions (m×n×o)
                    # m = num_simulations, n = num_experiments, o = length of harmonics
                    harmonics_list = []
                    
                    # First, calculate harmonics for the actual data
                    data_harmonics = np.array([
                        np.abs(sci.plot.generate_harmonics(t, i, hanning=True, one_sided=True, harmonics=self.all_harmonics))
                        for t, i in zip(all_times, all_data)
                    ])
                    harmonics_list.append(data_harmonics)
                    
                    # Then calculate harmonics for each simulation
                    for q in range(0, len(all_simulations)):
                        sim = all_simulations[q]
                        sim_harmonics = np.array([
                            np.abs(sci.plot.generate_harmonics(t, i, hanning=True, one_sided=True, harmonics=self.all_harmonics))
                            for t, i in zip(all_times, sim)
                        ])
                        harmonics_list.append(sim_harmonics)
                    
                    
                    plot_harmonics = np.array(harmonics_list)
                    
                    # Create lists for styling parameters
                    alphas = [1] + [0.75] * len(all_simulations)
                    
                    line_styles = ["-"] + [linestyles[l % len(linestyles)] for l in range(0, len(all_simulations))]
                    if kwargs["sim_plot_options"]=="simple":
                        colors = [None] + ["black"] * len(all_simulations)#
                        patheffects=[None]*(len(all_simulations)+1)
                    else:
                        colors=[None]
                        patheffects=[None]
                        for r in range(0, len(plot_colours)):
                            colors+=[plot_colours[r]]
                            patheffects+=[path_effects]
                    dmax=np.max(np.array(data_harmonics), axis=None)        
                    scaled_harmonics=self.process_harmonics(plot_harmonics, additional_maximum=dmax)
                    axis, line_list=self.plot_scaled_harmonics(ax, scaled_harmonics,
                                                alpha=alphas, 
                                                linestyle=line_styles, 
                                                colour=colors, 
                                                label_list=label_list,
                                                scale=True,
                                                patheffects=patheffects)
                    if kwargs["interactive_mode"]==True:
                        self.simulation_plots[groupkey]=line_list#lines are stored columnwise per plot and rowise per harmonics
                        self.simulation_plots["maxima"][groupkey]=dmax
                        self.simulation_plots["data_harmonics"][groupkey]=data_harmonics
                if kwargs["show_legend"]==True:
                    self.add_legend(ax, groupkey)
                if kwargs["simulation_labels"] is not None:
                   
                    if i%2==0 and i//2==len(self.grouping_keys)//4:
                        twinx=ax.twinx()
                        twinx.set_yticks([])
                        ylim=ax.get_ylim()
                        xlim=ax.get_xlim()
                        for r in range(0, len(kwargs["simulation_labels"])):
                            if kwargs["sim_plot_options"]=="simple":
                                twinx.plot(xlim[0],ylim[0], color="black", linestyle=linestyles[r%4])
                            else:
                                twinx.plot(xlim[0],ylim[0], color=plot_colours[r], linestyle=linestyles[r%4], path_effects=path_effects, label=kwargs["simulation_labels"][r])
                        twinx.legend(ncols=len(kwargs["simulation_labels"]), bbox_to_anchor=[0.5, -0.1], loc="center")
        fig=plt.gcf()                                                                  
        fig.set_size_inches(16, 10)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        if kwargs["interactive_mode"]==False:
            if kwargs["savename"] is not None:
                fig.savefig(kwargs["savename"], dpi=500)
                plt.close()
            else:
                plt.show()
    def interactive_front_results(self, all_simulations, target_key, sim_address, score_address):
        fig = plt.figure()
        

        if len(self.grouping_keys)%2!=0:
            num_cols=int(len(self.grouping_keys)+1)/2
        else:
            num_cols=int(len(self.grouping_keys)/2)
        spec = fig.add_gridspec(2, num_cols+1)
        axes=[[0 for x in range(0, num_cols)] for y in range(0, 2)]
        for i in range(0, 2):
            for j in range(0, num_cols):
                axes[i][j]=fig.add_subplot(spec[i,j])
        if len(self.grouping_keys)%2!=0:  
            axes[1][-2].set_axis_off()
        score_plt=fig.add_subplot(spec[0,-1])
        axes=np.array(axes)
        all_simulation_traces=[x["saved_simulations"] for x in all_simulations]
        all_scores=[x["scores"] for x in all_simulations]
        x_scores=np.array([x[target_key[0]] for x in all_scores])
        x_idx=np.argsort(x_scores)
        x_scores=x_scores[x_idx]
        y_scores=np.array([x[target_key[1]] for x in all_scores])[x_idx]
        score_plt.scatter(x_scores, y_scores, picker=True)
        score_plt.set_xlabel(target_key[0])
        score_plt.set_ylabel(target_key[1])
        score_plt.set_xscale("log")
        score_plt.set_yscale("log")
        self.scat=score_plt.scatter(x_scores[0], y_scores[0], c="darkred", edgecolors="black")
        
        self.plot_dict={"traces":all_simulation_traces, "scores":[x_scores, y_scores], "indexes":x_idx}
        
        

        self.results(all_simulation_traces[x_idx[0]], savename=None, pre_saved=True, target_key=target_key, sim_plot_options="simple", axes=axes, interactive_mode=True)
        
        self.interactive_fig=fig
        def update_plots(event):
            
            new_idx=event.ind[0]
            
            idx=self.plot_dict["indexes"][new_idx]
            get_data=self.plot_dict["traces"][idx]
            grouped_data=self.process_simulation_dict([get_data], pre_saved=True)[0]
            for i in range(0, len(self.grouping_keys)):
                groupkey=self.grouping_keys[i]
                if "ft" not in groupkey:
                    for j in range(0, len(grouped_data[groupkey])):
                        self.simulation_plots[groupkey][j].set_ydata(grouped_data[groupkey][j])
                else:
                    all_times=[self.classes[x]["times"] for x in self.experiment_grouping[groupkey]]
                    
                    sim_harmonics = np.array([
                        np.abs(sci.plot.generate_harmonics(t, i, hanning=True, one_sided=True, harmonics=self.all_harmonics))
                        for t, i in zip(all_times, [self.scale(get_data[x], groupkey, x) for x in self.experiment_grouping[groupkey]])
                    ])
                    plot_harmonics=np.array([self.simulation_plots["data_harmonics"][groupkey], sim_harmonics])
                    scaled_harmonics=np.array(self.process_harmonics(plot_harmonics, additional_maximum=self.simulation_plots["maxima"][groupkey])["scaled_data"])
                    existing_harmonic_plots=self.simulation_plots[groupkey]
                    dimensions=scaled_harmonics.shape
                    #print(dimensions)
                    num_harms=dimensions[1]
                    num_plots=dimensions[0]     
                    biggest=0
                    smallest=1e23
                    for k in range(0, num_plots):#column
                        for j in range(0, num_harms):#row (total_harmonic_plots)
                        
                            
                            current_harmonic=scaled_harmonics[k, j, 1, 1, :]
                            biggest=max(max(current_harmonic), biggest)
                            smallest=min(min(current_harmonic), smallest)
                            plotdx=(k*num_harms)+j
                            #print(plotdx)
                            self.simulation_plots[groupkey][plotdx][0].set_ydata(scaled_harmonics[k, j, 0, 1, :])
                            self.simulation_plots[groupkey][plotdx][1].set_ydata(current_harmonic)
                    #print(np.max(scaled_harmonics, axis=None))
                    axes[i%2, i//2].set_ylim([0.95*smallest, 1.05*biggest])   
                    #scaled data is returned as experiment -> harmonic -> plot
                    #lines are stored columnwise per plot and rowise per harmonics
            x=self.plot_dict["scores"][0][new_idx]
            y=self.plot_dict["scores"][1][new_idx]
            self.scat.set_offsets([x, y])
            self.interactive_fig.canvas.draw_idle()
        
        self.interactive_fig.canvas.mpl_connect("pick_event", update_plots)



        plt.show()
    def ax_results_extraction(self, dataloc, num_sets, saveloc, client_name="ax_client.npy"):
        results={key:[] for key in self.grouping_keys}
        linestart=len(r"Parameterization:</em><br>")

        for m in range(0, num_sets):
            ax_client=np.load("{0}/set_{1}/{2}".format(dataloc,m, client_name), allow_pickle=True).item()["saved_frontier"]
            for z in range(0, len(self.grouping_keys)):
                values=ax_client.get_contour_plot(param_x="k0", param_y="alpha", metric_name=self.grouping_keys[z])
                arms=values[0]["data"][2]["text"]
                value_dict={}
                for i in range(0, len(arms)):
                    key="arm{0}".format(i+1)
                    first_split=arms[i].split("<br>")
                    value_dict[key]={}
                    value_dict[key]["score"]=float(first_split[1][first_split[1].index(": ")+1:first_split[1].index("(")])
                    
                    for j in range(2, len(first_split)):
                        param_split=first_split[j].split(": ")
                        value_dict[key][param_split[0]]=float(param_split[1])
                    results[self.grouping_keys[z]].append(value_dict)
        np.save(saveloc, results)
        return results
    def sort_results(self, results_dict):
        keyr=list(results_dict.keys())
        best_params={key:{} for key in self.grouping_keys}
        all_scores={key:[] for key in self.grouping_keys}
        for key in keyr:
            bestscore=1e23

            for i in range(0, len(results_dict[key])):
                for arm in results_dict[key][i].keys():
                    all_scores[key].append(results_dict[key][i][arm]["score"])
                    if results_dict[key][i][arm]["score"]<0:
                        continue
                    if results_dict[key][i][arm]["score"]<bestscore:
                        bestscore=results_dict[key][i][arm]["score"]
                        best_params[key]["score"]=bestscore
                        best_params[key]["params"]=[results_dict[key][i][arm][key2] for key2 in self.all_parameters]
                        best_params[key]["arm"]=arm      
        return best_params
    def results_table(self, parameters, mode="table"):
        simulation_values=self.parse_input(parameters)
        un_normed_values={}
        l_optim_list=0
        for classkey in self.class_keys:
            
            cls=self.classes[classkey]["class"]
            current_len=max(len(cls.optim_list), l_optim_list)
            if current_len>l_optim_list:
                l_optim_list=current_len
                longest_list=cls.optim_list
            normed_params_list=simulation_values[classkey]
            un_normed_values[classkey]=dict(zip(cls.optim_list, cls.change_normalisation_group(normed_params_list, "un_norm")))
            if mode=="simulation":
                print(classkey)
                print(un_normed_values)
        if mode=="simulation":
            return
        for classkey in self.class_keys:
            cls=self.classes[classkey]["class"]
            for param in cls.optim_list:
                if param not in longest_list:
                    longest_list+=[param]
        header_list=["Parameter"]+longest_list
        table_data=[
            [classkey]+[sci._utils.format_values(un_normed_values[classkey][x],3)+","
                if x in un_normed_values[classkey] else "*"
                for x in longest_list]
            for classkey in self.class_keys
        ]
        table=tabulate.tabulate(table_data, headers=header_list, tablefmt="grid")
        return table    
    def open_pareto(self, file, **kwargs):
        if "key" not in kwargs:
            kwargs["key"]="frontier"
        if "skip_negatives" not in kwargs:
            kwargs["skip_negatives"]=True
        try:
            frontier=np.load(file, allow_pickle=True).item()[kwargs["key"]]
        except KeyError as err:
            raise KeyError("Frontier file dictionary doesn't contain key ({0})".format(err))
        except Exception as err:
            print("Some other error with regards to loading the pareto file {0}".format(err))
            raise 
        points=[]
        # For each point on the Pareto frontier
        for q in range(len(frontier.param_dicts)):
            point = {
                "parameters": frontier.param_dicts[q],
                "scores": {}
            }
            
            # Add all metric scores for this point
            for metric in frontier.means.keys():
                if kwargs["skip_negatives"]==True:
                    if frontier.means[metric][q]<0:
                        return file, None, None
                point["scores"][metric] = frontier.means[metric][q]
                
            points.append(point)
        return points, frontier.primary_metric, frontier.secondary_metric
    def extract_scores(self, points, keyx, keyy):
        paretox=[x["scores"][keyx] for x in points]
        paretoy=[y["scores"][keyy] for y in points]
        return np.array(paretox), np.array(paretoy)
    def get_pareto_points(self, points, keyx,keyy, s=1, **kwargs):
        if "get_knee_region" not in kwargs:
            kwargs["get_knee_region"]=False
        x, y=self.extract_scores(points, keyx, keyy)
        knee=KneeLocator(x, y, curve="convex", direction="decreasing", online=False, S=s)

        values={"knee":{"x":knee.knee, "y":knee.knee_y, "keyx":keyx, "keyy":keyy}}
        values["minx"]={"x":min(x), "y":y[np.where(x==min(x))][0]}
        values["miny"]={"x":x[np.where(y==min(y))][0], "y":min(y)}
        for key in values:
            xidx=np.where(x==values[key]["x"])
            yidx=np.where(y==values[key]["y"])  
            if len(xidx)>1 or len(yidx)>1:
                common_idx=set(xidx[0]).intersection(set(y_idx[0]))[0]
            else:

                common_idx=xidx[0][0]
            values[key]["index"]=common_idx
            
            values[key]["parameters"]=points[common_idx]["parameters"]
            values[key]["other_scores"]=points[common_idx]["scores"]
        
        if kwargs["get_knee_region"] is not False:
            idx=values["knee"]["index"]
            if isinstance(kwargs["get_knee_region"], list) is False:
                if isinstance(kwargs["get_knee_region"], int) is True:
                    kwargs["get_knee_region"]=[kwargs["get_knee_region"]]
                else:
                    raise ValueError("get_knee_region either needs to be int or list of ints, not {0}".format(type(kwargs["get_knee_region"])))
            for i in range(0, len(kwargs["get_knee_region"])):
                value=kwargs["get_knee_region"][i]
                for sign in [-1, 1]:
                    if sign==-1:
                        key="knee_minus_{0}".format(value)
                    else:
                        key="knee_plus_{0}".format(value)
                    curr_idx=idx+sign*value
                    if curr_idx<0:
                        curr_idx=0
                    elif curr_idx>len(x)-1:
                        curr_idx=len(x)-1
                    values[key]={
                        "x":x[curr_idx],
                        "y":y[curr_idx],
                        "index":curr_idx,
                        "parameters":points[curr_idx]["parameters"],
                        "other_scores":points[curr_idx]["scores"]
                    }
        return values

    def plot_front(self, opened_file, keyx, keyy, **kwargs):
        if "ax" not in kwargs:
            kwargs["ax"]=None
        if "colour" not in kwargs:
            kwargs["colour"]=None
        if "plot_knee" not in kwargs:
            kwargs["plot_knee"]=False
        if "plot_negatives" not in kwargs:
            kwargs["plot_negatives"]=False
        if kwargs["ax"] is None:
            fig,ax=plt.subplots()
        else:
            ax=kwargs["ax"]
        points=opened_file
        
        xaxis, yaxis=self.extract_scores(points,keyx, keyy)
        if kwargs["plot_negatives"]==False:
                found_neg=False
                for axis_plot in [xaxis, yaxis]:
                    if any([x<0 for x in axis_plot]):
                        found_neg=True
                        return
        ax.scatter(xaxis, yaxis, c=kwargs["colour"])
        found_neg=False
        if kwargs["plot_knee"] is True:
            ppoints=self.get_pareto_points(points, keyx, keyy)
            ax.scatter(ppoints["knee"]["x"], ppoints["knee"]["y"], s=50, edgecolors="black")
    def plot_all_fronts(self, file_list, **kwargs):
        if "keylabel" not in kwargs:
            kwargs["keylabel"]=True
        if kwargs["keylabel"]==True:
            letterdict=dict(zip(self.grouping_keys, ascii_uppercase[:len(self.grouping_keys)]))
            legend_dict={key:"{0} : {1}".format(letterdict[key], key) for key in self.grouping_keys}
        num_metrics=len(self.grouping_keys)
        if "plot_knee" not in kwargs:
            kwargs["plot_knee"]=False
        if "axes" not in kwargs: 
            kwargs["axes"]=None
        if kwargs["axes"] is None:
            fig, ax=plt.subplots(num_metrics, num_metrics)
        else:
            if len(kwargs["axes"])!=num_metrics:
                raise ValueError("Subplots need to have {0} columns not {1}".format(num_metrics, len(kwargs["ax"])))   
            ax=kwargs["axes"]    
        if "colours" not in kwargs:
            kwargs["colours"]=sci._utils.colours
        if "plot_negatives" not in kwargs:
            kwargs["plot_negatives"]=False
        
         
        if isinstance(file_list, str):
            files=[os.path.join(file_list, x) for x in os.listdir(file_list)]
        else:
            files=file_list
        for q in range(0, len(files)):
            file=files[q]
            opened_file, key1, key2=self.open_pareto(file, skip_negatives=False)
            for i in range(0, num_metrics):
                for j in range(0, num_metrics):
                    if i>j:
                        if self.grouping_keys[i] in [key1, key2] and self.grouping_keys[j] in [key1, key2]:
                            self.plot_front(opened_file, self.grouping_keys[j], self.grouping_keys[i], 
                                                        colour=kwargs["colours"][q%len(kwargs["colours"])], 
                                                        plot_knee=kwargs["plot_knee"], 
                                                        plot_negatives=kwargs["plot_negatives"],
                                                        ax=ax[i,j])
        for i in range(0, num_metrics):
            for j in range(0, num_metrics):
                if i<=j:
                    ax[i,j].set_axis_off()
                else:
                    if i==num_metrics-1:
                        if kwargs["keylabel"]==True:
                            xlabel=letterdict[self.grouping_keys[j]]
                        else:
                            xlabel=self.grouping_keys[j]
                        ax[i,j].set_xlabel(xlabel)
                    if j==0:
                        if kwargs["keylabel"]==True:
                            ylabel=letterdict[self.grouping_keys[i]]
                        else:
                            ylabel=self.grouping_keys[i]
                        ax[i,j].set_ylabel(ylabel)
        if kwargs["keylabel"]==True:
            for i in range(0, num_metrics):

                ax[0,-1].plot(0, 0, label=legend_dict[self.grouping_keys[i]])
            ax[0,-1].legend(loc="center", handlelength=0)
    def process_pareto_directories(self, loc, filekey="frontier"):
        split_keys=list(itertools.combinations(self.grouping_keys, 2))
        combo_keys=["{0}&{1}".format(x[0], x[1]) for x in split_keys]
        split_key_addresses=dict(zip(combo_keys, split_keys))
        all_points = {key:[] for key in combo_keys}
        skipped_files=[]
        read_files=[]
        def search_directories(current_path):
            items = os.listdir(current_path)
            directories = []
            files = []
            
            # Separate files and directories
            for item in items:
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path):
                    directories.append(item_path)
                else:
                    files.append(item_path)
            
            # If we have directories, keep searching deeper
            if directories:
                for directory in directories:
                    search_directories(directory)
            # If we only have files, process them
            elif files:
                for file in files:
                    try:
                        points, keyx, keyy= self.open_pareto(file, skip_negatives=True, key=filekey)


                        if isinstance(points, str) is False:
                            read_files.append(file)
                            for key in combo_keys:
                                if keyx in split_key_addresses[key] and keyy in split_key_addresses[key]:
                                    all_points[key]+=points
                        else:
                            skipped_files.append(points)
                    except Exception as e:
                        # Handle cases where get_knees can't operate on the file
                        print(f"Couldn't process file {file}: {e}")
                        raise
        
        # Start the recursive search
        search_directories(loc)
        total_len=0
        if skipped_files:
            print("Skipped {0} files (of {1}) due to negative fronts when searching for knees".format(len(skipped_files), len(read_files)+len(skipped_files)))
        return all_points
                    


                                
                                    
                                    


        
                    





      
            

                





            
