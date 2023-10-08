Label = {0: 'a hand gesture of pointing with index finger ',
         1: 'a hand entending all five fingers',
         2: 'a clenched fist',
         3: 'a hand gesture of ok',
         4: 'a hand gesture of finger heart',
         5: 'a hand gesture of V sign',
         6: 'a hand gesture of three-finger salute',
         7: 'a hand gesture of four fingers up',
         8: 'a hand gesture of shaka sign',
         9: 'a hand gesture of the sign of the horns',
         10: 'a hand gesture of finger gun',
         11: 'a hand gesture of thumbs up',
         12: 'a hand gesture of bent index finger',
         13: 'a hand gesture of showing little finger'}

refined_Label = {}
for k, v in Label.items():
    if len(v.split(",")) == 1:
        select = v
    else:
        select = v.split(",")[0]
    refined_Label.update({k: select})
