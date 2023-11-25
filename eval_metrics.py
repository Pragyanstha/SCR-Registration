import json
from argparse import ArgumentParser

import numpy as np


def rearrange(metrics, th=None):
    mTREs = []
    err_Rs = []
    err_Ts = []
    ids = []
    elts = []
    for k, v in metrics.items():
        if np.isnan(v["mTRE"]):
            mTREs.append(np.nan)
            err_Rs.append(np.nan)
            err_Ts.append(np.nan) 
        else:
            mTREs.append(v["mTRE"])
            err_Rs.append(v["rotation_error"])
            err_Ts.append(v["translation_error"])
        ids.append(k)
        try:
            elts.append(v["time"])
        except:
            elts.append(np.nan)

    mTREs = np.array(mTREs)
    err_Rs = np.array(err_Rs)
    err_Ts = np.array(err_Ts)
    elts = np.array(elts)

    if th is not None:
        mask = mTREs > th
        mTREs[mask] = np.nan
        err_Rs[mask] = np.nan
        err_Ts[mask] = np.nan

    nums = np.count_nonzero(~np.isnan(mTREs))
    return {
        "ids": ids,
        "mTRE [mm]": mTREs,
        "Rotation Error [deg]": err_Rs,
        "Translation Error [mm]": err_Ts,
        "Number of samples": nums,
        "times": elts
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resultdir", type=str, required=True)
    args = parser.parse_args()

    RESULT_PATH = args.resultdir + "/metrics.json"
    with open(RESULT_PATH, "r") as f:
        temp = json.load(f)
        result_data = rearrange(temp)

    mTRE_25th = np.nanpercentile(result_data["mTRE [mm]"], 25)
    mTRE_50th = np.nanpercentile(result_data["mTRE [mm]"], 50)
    mTRE_95th = np.nanpercentile(result_data["mTRE [mm]"], 95)
    mTRE_mean = np.nanmean(result_data["mTRE [mm]"])
    mTRE_std = np.nanstd(result_data["mTRE [mm]"])
    exec_time = np.nanmean(result_data["times"])
    GFR = np.count_nonzero(result_data["mTRE [mm]"] > 10) / result_data["mTRE [mm]"].shape[0]*100
    print(f"mTRE_25th: {mTRE_25th:.3f}")
    print(f"mTRE_50th: {mTRE_50th:.3f}")
    print(f"mTRE_95th: {mTRE_95th:.3f}")
    print(f"mTRE_mean: {mTRE_mean:.3f}")
    print(f"mTRE_std: {mTRE_std:.3f}")
    print(f"exec_time: {exec_time:.3f}")
    print(f"GFR: {GFR:.3f}")
