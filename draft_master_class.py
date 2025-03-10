import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from scipy.interpolate import CubicSpline
import itertools
import Surface_confined_inference as sci
from pathlib import Path
import tabulate
import re
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
            parameter_list=self.parameter_map[group_key]
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
    def check_grouping(self,show_legend=False):
        if len(self.grouping_keys)%2!=0:
            num_cols=(len(self.grouping_keys)+1)/2
            fig,axes=plt.subplots(2, int(num_cols))
            axes[1, -1].set_axis_off()
        else:
            num_cols=len(self.grouping_keys)/2
            fig,axes=plt.subplots(2, int(num_cols))
        for i in range(0, len(self.grouping_keys)):
            groupkey=self.grouping_keys[i]
            ax=axes[i%2, i//2]
            ax.set_title(groupkey, fontsize=8)
            all_data=[self.scale(self.classes[x]["data"], groupkey, x) for x in self.experiment_grouping[groupkey]]
            all_times=[self.classes[x]["times"] for x in self.experiment_grouping[groupkey]]
            all_zeros=[self.scale(self.classes[x]["zero_sim"], groupkey, x) for x in self.experiment_grouping[groupkey]]
            label_list=[",".join(x.split("-")[1:]) for x in self.experiment_grouping[groupkey]]
            if "type:ft" not in groupkey:
                self.plot_stacked_time(ax, all_data, label_list=label_list)
                self.plot_stacked_time(ax, all_zeros, alpha=0.75, linestyle="--", colour="black")
            else:
                data_harmonics=[
                    np.abs(sci.plot.generate_harmonics(t, i, hanning=True, one_sided=True, harmonics=self.all_harmonics))
                    for t,i in zip(all_times, all_data, )
                ]
                zero_harmonics=[
                    np.abs(sci.plot.generate_harmonics(t, i, hanning=True, one_sided=True, harmonics=self.all_harmonics))
                    for t,i in zip(all_times, all_zeros, )
                ]
                self.plot_stacked_harmonics(ax, [data_harmonics, zero_harmonics], alpha=[1,0.75], 
                                                                                linestyle=["-", "--"], 
                                                                                colour=[None, "black"], 
                                                                                label_list=label_list,
                                                                                scale=True)

            if show_legend==True:
                self.add_legend(ax, groupkey)

        fig.set_size_inches(16, 12)
        plt.tight_layout()
        plt.show()
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
        current_len=0
        for i in range(0, len(data_list)):
            xaxis=range(current_len, current_len+len(data_list[i]))
            if kwargs["label_list"] is not None:
                label=kwargs["label_list"][i]
            else:
                label=None
            axis.plot(xaxis, data_list[i], label=label, alpha=kwargs["alpha"], linestyle=kwargs["linestyle"], color=kwargs["colour"])
            current_len+=len(data_list[i])
        axis.set_xticks([])
        return axis
    def plot_stacked_harmonics(self, axis,harmonics_list, **kwargs):
        if "colour" not in kwargs:
            kwargs["colour"]=[None]
        if "linestyle" not in kwargs:
            kwargs["linestyle"]=["-"]
        if "alpha" not in kwargs:
            kwargs["alpha"]=[1]
        if "label_list" not in kwargs:
            kwargs["label_list"]=None
        if "harmonics" not in kwargs:
            kwargs["harmonics"]=self.all_harmonics
        if "scale" not in kwargs:
            kwargs["scale"]=True
        num_harmonics=len(kwargs["harmonics"])
        arrayed=np.array(harmonics_list)
        maximum=np.max(arrayed, axis=None)
        num_plots=arrayed.shape[0]
        num_experiments=arrayed.shape[1]
        for key in ["colour", "linestyle", "alpha"]:
            if isinstance(kwargs[key], list) is False:
                raise ValueError("{0} needs to be wrapped into a list".format(key))
            if len(kwargs[key])!=num_plots:
                if len(kwargs[key])==1:
                    kwargs[key]=[kwargs[key][0] for x in range(0, num_plots)]
                else:
                    raise ValueError("{0} needs to be the same length as the number of plots".format(key))
        if "residual" not in kwargs:
            kwargs["residual"]=False
        if kwargs["residual"]==True and num_plots!=2:
            raise ValueError("Can only do two sets of harmonics for a residual plot")
        current_len=0
        
        for m in range(0, num_experiments):
            for i in range(0, num_harmonics):
                current_maximum=np.max(np.array([arrayed[x][m][i,:] for x in range(0, num_plots)]), axis=None)
                offset=(num_harmonics-i)*1.1*maximum
                ratio=maximum/current_maximum
                if kwargs["scale"]==False:
                        ratio=1
                for j in range(0, num_plots):
                    
                    xaxis=range(current_len, current_len+len(arrayed[j][m][i,:]))
                    if i==0 and j==0 and kwargs["label_list"] is not None:
                        label=kwargs["label_list"][m]
                    else:
                        label=None
                    if kwargs["colour"][j]== None:
                        colour=sci._utils.colours[m]
                    else:
                        colour=kwargs["colour"][j]
                    
                    axis.plot(xaxis, ratio*arrayed[j][m][i,:]+offset, label=label, alpha=kwargs["alpha"][j], linestyle=kwargs["linestyle"][j], color=colour)
            current_len+=len(arrayed[j][m][i,:])
        axis.set_xticks([])
    def results(self, parameters, **kwargs):
        if "target_key" not in kwargs:
            target_key=None
        else:
            target_key=kwargs["target_key"]
        if "savename" not in kwargs:
            kwargs["savename"]=None
        if "show_legend" not in kwargs:
            kwargs["show_legend"]=False
        if len(self.grouping_keys)%2!=0:
            num_cols=(len(self.grouping_keys)+1)/2
            fig,axes=plt.subplots(2, int(num_cols))
            axes[1, -1].set_axis_off()
        else:
            num_cols=len(self.grouping_keys)/2
            fig,axes=plt.subplots(2, int(num_cols))
        simulation_values=self.evaluate(parameters)
        for i in range(0, len(self.grouping_keys)):
            groupkey=self.grouping_keys[i]
            ax=axes[i%2, i//2]
            
            if groupkey==target_key:
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(4)
                    ax.tick_params(width=4)
            ax.set_title(groupkey, fontsize=8)
            all_data=[self.scale(self.classes[x]["data"], groupkey, x) for x in self.experiment_grouping[groupkey]]
            all_times=[self.classes[x]["times"] for x in self.experiment_grouping[groupkey]]
            all_simulations=[self.scale(simulation_values[x], groupkey, x) for x in self.experiment_grouping[groupkey]]
            label_list=[",".join(x.split("-")[1:]) for x in self.experiment_grouping[groupkey]]
            if "type:ft" not in groupkey:
                self.plot_stacked_time(ax, all_data, label_list=label_list)
                self.plot_stacked_time(ax, all_simulations, alpha=0.75, linestyle="--", colour="black")
            else:
                data_harmonics=[
                    np.abs(sci.plot.generate_harmonics(t, i, hanning=True, one_sided=True, harmonics=self.all_harmonics))
                    for t,i in zip(all_times, all_data, )
                ]
                sim_harmonics=[
                    np.abs(sci.plot.generate_harmonics(t, i, hanning=True, one_sided=True, harmonics=self.all_harmonics))
                    for t,i in zip(all_times, all_simulations, )
                ]
                self.plot_stacked_harmonics(ax, [data_harmonics, sim_harmonics], alpha=[1,0.75], 
                                                                                linestyle=["-", "--"], 
                                                                                colour=[None, "black"], 
                                                                                label_list=label_list,
                                                                                scale=True)
            if kwargs["show_legend"]==True:
                self.add_legend(ax, groupkey)
                                                                            
        fig.set_size_inches(16, 12)

        plt.tight_layout()
        if kwargs["savename"] is not None:
            fig.savefig(kwargs["savename"], dpi=500)
            plt.close()
        else:
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
        print(table)
            

            




            