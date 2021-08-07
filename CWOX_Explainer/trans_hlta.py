def replace(nodes):
    for node in nodes:
        node["text"] = list(map(lambda x: x.split("-")[0], node["text"].split(" ")))
        replace(node["children"])

with open("fullname_.json", "r") as f: x = json.load(x)
replace(x)
with open("fullname_.json", "w") as f: json.dump(x, f)

