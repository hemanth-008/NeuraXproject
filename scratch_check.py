import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
costs = config.get("economics", {})
print("C_ESCAPE type:", type(costs.get("c_escape", 50.0)))
print("C_ESCAPE value:", costs.get("c_escape", 50.0))
