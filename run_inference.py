import os
import argparse
import yaml
import json

import numpy as np

from utils.data_loading import DFLDataset
from utils.evaluations import  evaluate_scrnet, evaluate_posenet, evaluate_dflnet
from utils.misc import NumpyEncoder
from models import get_model


def main(config):
    BASE_OUTDIR = config["resultdir"]
    os.makedirs(BASE_OUTDIR, exist_ok=True)
    PRED_OUTDIR = os.path.join(BASE_OUTDIR, "predictions")
    WEIGHT_PATH = os.path.join(config["weightpath"])
    model_cfg = config["model"]
    model = get_model(config["modelname"]).load_from_checkpoint(WEIGHT_PATH, model_cfg=model_cfg)
    model.eval()
    dataset = DFLDataset(config["datadir"], "test", model_cfg["pad"], model_cfg["heatmap"] if "heatmap" in model_cfg else False, select_first=config["select_first"] if "select_first" in config else None)

    mesh = dataset.get_mesh()
    num_dataset = len(dataset)
    print(f"Testing {num_dataset} images")
    with open(os.path.join(BASE_OUTDIR, "configs.yaml"), "w") as f:
        yaml.dump(config, f)

    results = {}
    for idx in range(num_dataset):
        data = dataset[idx]
        fileid = data["id"]
        
        gt_corr_entry = data["corr_entry"].numpy().transpose([1,2,0])
        # If visible target area is less than 10%, skip the image
        vis_ratio = np.count_nonzero(np.any(gt_corr_entry > 0, axis=-1))/(512*512)
        print(vis_ratio)
        if (vis_ratio < 0.10):
            print(f"Skipping id : {fileid}")
            continue

        OUTDIR = os.path.join(PRED_OUTDIR, fileid) 
        os.makedirs(OUTDIR, exist_ok=True)
        if config["modelname"] == "scrnet":
            result = evaluate_scrnet(data, model, OUTDIR, config, mesh=mesh, idx=idx)
        elif config["modelname"] == "posenet":
            result = evaluate_posenet(data, model, OUTDIR, config, mesh=mesh, idx=idx)
        elif config["modelname"] == "dflnet":
            result = evaluate_dflnet(data, model, OUTDIR, config, mesh=mesh, idx=idx)

        results[fileid] = result

    with open(os.path.join(BASE_OUTDIR, "metrics.json"), "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as yml:
        config = yaml.safe_load(yml)
    print(config)
    main(config)
