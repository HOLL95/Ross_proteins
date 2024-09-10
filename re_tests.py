import re

regex = r"(-)?(\d+\.)?\d+(e(\+|-)\d+)?(?=,)"

test_str = "1,    -0.4339259516, 0.0547731608,     2.2089585444e+03, 6.3258601640e-11,         2.9167288598e+03, 3.7301741496e-05, 1.9479999647e-05,  1.9912540553e-05,  9.8413624335e-07,  8.3687923956,  0.4166977967, 220.8342602052, "

matches = re.finditer(regex, test_str, re.MULTILINE)

for matchNum, match in enumerate(matches, start=1):
    
    print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
    
    for groupNum in range(0, len(match.groups())):
        groupNum = groupNum + 1
        
        print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))
