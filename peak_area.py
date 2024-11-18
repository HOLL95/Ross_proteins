import Surface_confined_inference as sci
file="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Trumpet/DCV_m4D2_PGE_30_mV_s-1_5_deg.txt"
file="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Trumpet/DCV_m4D2_PGE_5000_mV_s-1_5_deg.txt"
sci.HeuristicMethod(file, method="DCVGamma", area=0.036)