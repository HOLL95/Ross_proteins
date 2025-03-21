 def pareto_explorer(self, fileloc,key1, key2, **kwargs):
        if "savename" not in kwargs:
            kwargs["savename"]=None
        if "show_legend" not in kwargs:
            kwargs["show_legend"]=True
        if "explore" not in kwargs:
            kwargs["explore"]="wide"
        if "filter_level" not in kwargs:
            kwargs["filter_level"]="ranking"
        
        elif kwargs["filter_level"] not in ["most_winners", "ranking"]:
            raise ValueError("The only allowed values for filter_level are `most_winners` or `ranking`, not {0}".format(kwargs["filter_level"]))

        elif isinstance(kwargs["explore"], int ) is False:
            raise ValueError("explore_mode must either be `wide` or of type int")
        if kwargs["explore"]=="wide":
            all_knees=self.open_knees(loc=fileloc)
            explore_keys=["minx", "knee","miny"]
        else:
            target_key="knee_region_{0}".format(kwargs["explore"])
            all_knees=self.open_knees(loc=fileloc, keyarg=target_key)
            explore_keys=["knee_minus_{0}".format(kwargs["explore"]), "knee", "knee_plus_{0}".format(kwargs["explore"])]
        if "labels" not in kwargs:
            kwargs["labels"]=explore_keys
        all_parameters=[]
        for i in range(0, len(explore_keys)):
            filtered_knees=self.filter_knees(knees=copy.deepcopy(all_knees), key=explore_keys[i], filter_level=kwargs["filter_level"])
            for key_combo in filtered_knees.keys():
                target_keys=key_combo.split("&")
                if key1 in target_keys and key2 in target_keys:
                    print(filtered_knees[key_combo]["scores"])
                    parameters=[filtered_knees[key_combo]["parameters"][z] for z in self.all_parameters] 
                    all_parameters.append(parameters)
        self.results(all_parameters, savename=kwargs["savename"], show_legend=kwargs["show_legend"], target_key=[key1, key2],simulation_labels=kwargs["labels"])
    def process_knee_directories(self, loc, key=None):
        all_knees = []
        skipped_files=[]
        knee_counter=0
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
                        knees = self.get_knees(file, skip_negatives=True, key=key)
                        if isinstance(knees, str) is False:
                            all_knees.append(knees)
                        else:
                            skipped_files.append(knees)
                    except Exception as e:
                        # Handle cases where get_knees can't operate on the file
                        print(f"Couldn't process file {file}: {e}")
                        #raise
        
        # Start the recursive search
        search_directories(loc)
        if skipped_files:
            print("Skipped {0} files (of {1}) due to negative fronts when searching for knees".format(len(skipped_files), len(all_knees)+len(skipped_files)))
        return all_knees
   
    def open_knees(self, loc, keyarg=None):
        dirs=os.listdir(loc)
        all_knees=self.process_knee_directories(loc, key=keyarg)
        return all_knees
    def filter_knees(self,**kwargs):
        allowed_keys=["knee", "minx","miny"]
        if "key" not in kwargs:
            kwargs["key"]="knee"
        if kwargs["key"] in allowed_keys:
            keyarg=None
        elif "knee_minus" in kwargs["key"] or "knee_plus" in kwargs["key"]:
            value=kwargs["key"].split("_")[-1]
            keyarg="knee_region_{0}".format(value)
        else:
            raise ValueError("Key needs to be either in {0} or contain knee_minus_ or knee_plus_, not {1}".format(allowed_keys, kwargs["key"]))
        if "loc" not in kwargs:
            kwargs["loc"]=None
            if "knees" not in kwargs:
                raise ValueError("Either loc or knees need to be passed to function")
            else:
                all_knees=kwargs["knees"]
        if "knees" not in kwargs:
            kwargs["knees"]=None
            if "loc" not in kwargs:
                raise ValueError("Either loc or knees need to be passed to function")
            else:
                all_knees=self.open_knees(kwargs["loc"], keyarg=keyarg)
        
        if "filter_level" not in kwargs:
            filter_level="ranking"
        else:
            filter_level=kwargs["filter_level"]
        allowed_filters=["pairwise_dominated","other_score_dominated", "ranking", "most_winners"]
        if kwargs["filter_level"] not in allowed_filters:
            raise ValueError("filter_level arg `{0}` not in allowed options {1}".format(kwargs["filter_level"], allowed_filters))
        
        
        
        combinations=["{0}&{1}".format(x[0], x[1]) for x in itertools.combinations(self.grouping_keys, 2)]
        current_best={}
        for i in range(0, len(combinations)):
            keyx, keyy=combinations[i].split("&")
            for j in range(0, len(all_knees)):
                current_keys=[all_knees[j]["knee"][x] for x in ["keyx","keyy"]]
                if keyx in current_keys and keyy in current_keys:
                    if kwargs["key"] not in all_knees[j]:
                        raise ValueError("{1} not found in knee directory (only {0})".format(all_knees[j].keys(), kwargs["key"]))
                    if combinations[i] not in current_best:
                        
                        
                        current_best[combinations[i]]={
                                                        all_knees[j]["knee"]["keyx"]:[all_knees[j][kwargs["key"]]["x"]],
                                                        all_knees[j]["knee"]["keyy"]:[all_knees[j][kwargs["key"]]["y"]],
                                                        "parameters":[all_knees[j][kwargs["key"]]["parameters"]],
                                                        "scores":[all_knees[j][kwargs["key"]]["other_scores"]]
                                                        
                                                    }
                    else:
                        cb=current_best[combinations[i]]
                        proposed={
                                all_knees[j]["knee"]["keyx"]:all_knees[j][kwargs["key"]]["x"],
                                all_knees[j]["knee"]["keyy"]:all_knees[j][kwargs["key"]]["y"],
                                "parameters":all_knees[j][kwargs["key"]]["parameters"],
                                "scores":all_knees[j][kwargs["key"]]["other_scores"]
                                }
                        
                        dom_idx=[]
                        partial_idx=[]
                        for m in range(0, len(cb[keyy])):
                           
                            if proposed[keyx]<cb[keyx][m] and proposed[keyy]<cb[keyy][m]:
                                dom_idx.append(m)
                            elif proposed[keyx]>cb[keyx][m] and proposed[keyy]>cb[keyy][m]:
                                dom_idx=[]
                                break
                            elif proposed[keyx]<cb[keyx][m] or proposed[keyy]<cb[keyy][m]:
                                partial_idx.append(m)
                        
                        
                        if len(dom_idx)>=1:
                            current_len=len(current_best[combinations[i]][keyx])
                            for key in [keyx, keyy, "parameters", "scores"]:
                                current_best[combinations[i]][key]=[current_best[combinations[i]][key][x] for x in range(0, current_len) if x not in dom_idx]
                                current_best[combinations[i]][key].append(proposed[key])
                        elif len(partial_idx)>=1:
                            for key in [keyx, keyy, "parameters", "scores"]:
                                current_best[combinations[i]][key].append(proposed[key])
                    
        if filter_level=="pairwise_dominated":
            return current_best
        
        for i in range(0, len(combinations)):
            keyx, keyy=combinations[i].split("&")
            current_len=len(current_best[combinations[i]][keyy])
            removal_idx=[]
            current_os=[]
            for m in range(0, len(current_best[combinations[i]]["scores"])):
                scores=current_best[combinations[i]]["scores"][m]

                other_scores=[scores[x] for x in self.grouping_keys if x not in [keyx, keyy]]
                current_os.append(other_scores)
            dom_idx=[]
            iterations=list(itertools.combinations(range(0, len(current_os)), 2))
            for m in range(0, len(iterations)):
                idx1=iterations[m][0]
                idx2=iterations[m][1]
                if idx1 in dom_idx or idx2 in dom_idx:
                    continue
                score=[x>=y for x,y in zip(current_os[idx1], current_os[idx2])]

                if all(score):
                    dom_idx.append(idx1)
                elif not any(score):
                    dom_idx.append(idx2)

            for key in [keyx, keyy, "parameters", "scores"]:
                current_best[combinations[i]][key]=[current_best[combinations[i]][key][x] for x in range(0, current_len) if x not in dom_idx]
        if filter_level=="other_score_dominated":
            return current_best
        
        for i in range(0, len(combinations)):
            keyx, keyy=combinations[i].split("&")
            current_len=len(current_best[combinations[i]][keyy])
            
            
            ranks=[]
            for m in range(0, len(self.grouping_keys)):
                key=self.grouping_keys[m]
                get_scores=[current_best[combinations[i]]["scores"][x][key] for x in range(0, len(current_best[combinations[i]]["scores"]))]
                scorerank=np.add(np.argsort(get_scores), 1)
                ranks.append(scorerank)

            ranks=np.array(ranks)
            if filter_level=="ranking":
                colsum=np.sum(ranks, axis=0)
                #print(ranks)
                #print(colsum)
                windex=np.where(colsum==min(colsum))[0][0]
            elif filter_level=="most_winners":
                for g in range(0, len(ranks)):
                    ranks[g][np.where(ranks[g]>1)]=0
                colsum=np.sum(ranks, axis=0)
                #print(ranks)
                #print(colsum)
                windex=np.where(colsum==max(colsum))[0][0]
            for key in [keyx, keyy, "parameters", "scores"]:
                current_best[combinations[i]][key]=current_best[combinations[i]][key][windex]
                #print(current_best[combinations[i]][key])
        return current_best
            

#TODO
# simulation colours
# simulation labels
# different ways of extracting pareto front
#   
