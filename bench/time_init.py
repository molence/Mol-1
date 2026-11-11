import time, cProfile, pstats, sys
sys.path.insert(0, sys.argv[1])
sys.argv = sys.argv[:1]
from mol.config.base import configs as configs_base
from mol.config.data import data_configs
from mol.config.inference import inference_configs
from mol.config.model_type import model_configs
from mol.config.parser import parse_configs
from mol.model.mol import Mol
base = {**configs_base, "data": data_configs, **inference_configs}
def du(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict): du(d[k], v)
        else: d[k] = v
du(base, model_configs["mol-1-base"])
cfg = parse_configs(configs=base, arg_str="", fill_required_with_null=True)
t = time.time()
pr = cProfile.Profile(); pr.enable(); m = Mol(cfg); pr.disable()
print("init", time.time() - t)
pstats.Stats(pr).sort_stats("cumulative").print_stats(12)
