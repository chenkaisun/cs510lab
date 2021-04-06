import json
from IPython import embed

filepath = "data/allMeSH_2021.json"
instances = []

with open(filepath, encoding="ISO-8859-1") as f:
    for line in f:
        line = line.strip()
        if "meshMajor" in line:

            # comma
            ins = json.loads(line[:-1])

            # ignore empty label
            if len(ins["meshMajor"]):

                # remove irelevent item
                ins.pop("journal")
                ins.pop("title")
                ins.pop("year")
                instances.append(ins)
                print(ins)

print("Done")

'''
One example
{'abstractText': 'Leg muscles of the monkey ....', 'meshMajor': ['Muscles'], 'pmid': '14824396'}
'''
