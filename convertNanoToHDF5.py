#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import uproot
import numpy as np
import h5py

from utils import to_np_array


def convertNanoToHDF5(input_path, evt_name, maxevents=-1, is_data=False,):
    if input_path == '':
        sys.exit('Need to specify input and output files!')

    ##
    varList = [
            'nL1PuppiCands', 'L1PuppiCands_pt', 'L1PuppiCands_eta', 'L1PuppiCands_phi',
            'L1PuppiCands_charge', 'L1PuppiCands_pdgId', 'L1PuppiCands_puppiWeight'
    ]

    # event-level variables
    varList_mc = ['genMet_pt', 'genMet_phi',]
    l1met_list = ['ctl2METMet_pt', 'ctl2METMet_phi']
    cmssw_ml_list = ['DeepMETMet_pt', 'DeepMETMet_phi']

    d_encoding = {
        'L1PuppiCands_charge': {-999.0: 0,
                                -1.0: 1,
                                0.0: 2,
                                1.0: 3},
        'L1PuppiCands_pdgId_org': {-999.0: 0,  # NONE
                                -211.0: 1,  # HADMINUS
                                -130.0: 2,  # nothing
                                -22.0: 3,   # nothing
                                -13.0: 4,   # MUMINUS
                                -11.0: 5,   # ELEMINUS
                                11.0: 5,    # ELEPLUS
                                13.0: 4,    # MUPLUS
                                22.0: 3,    # PHOTON
                                130.0: 2,   # HADZERO
                                211.0: 1},  # HADPLUS

        'L1PuppiCands_pdgId': {-999.0: 0,  # NONE
                            130.0: 0,   # HADZERO
                            22.0: 1,    # PHOTON
                            -211.0: 2,  # HADMINUS
                            211.0: 3,   # HADPLUS
                            -11.0: 4,   # ELEMINUS
                            11.0: 5,    # ELEPLUS
                            -13.0: 6,   # MUMINUS
                            13.0: 7,    # MUPLUS
                            } 
    }


    if not is_data:
        varList = varList + varList_mc +l1met_list

    upfile = uproot.open(input_path)

    if maxevents > 0:
        tree = upfile["Events"].arrays(varList, entry_stop=maxevents)
    else:
        tree = upfile["Events"].arrays(varList)

    # general setup
    maxNPuppi = 128
    nFeatures = 9
    maxEntries = len(tree["nL1PuppiCands"])

    print(f"Found {maxEntries} Events")

    # L1 Puppi candidates
    X = np.zeros(shape=(maxEntries, maxNPuppi, nFeatures), dtype=float, order='F')

    pt = to_np_array(tree['L1PuppiCands_pt'], maxN=maxNPuppi)
    eta = to_np_array(tree['L1PuppiCands_eta'], maxN=maxNPuppi)
    phi = to_np_array(tree['L1PuppiCands_phi'], maxN=maxNPuppi)
    pdgid = to_np_array(tree['L1PuppiCands_pdgId'], maxN=maxNPuppi, pad=-999)
    charge = to_np_array(tree['L1PuppiCands_charge'], maxN=maxNPuppi, pad=-999)
    puppiw = to_np_array(tree['L1PuppiCands_puppiWeight'], maxN=maxNPuppi)

    X[:, :, 0] = pt
    X[:, :, 1] = pt * np.cos(phi) # px
    X[:, :, 2] = pt * np.sin(phi) # py
    X[:, :, 3] = eta
    X[:, :, 4] = phi
    X[:, :, 5] = puppiw

    # encoding
    X[:, :, 6] = np.vectorize(d_encoding['L1PuppiCands_pdgId'].__getitem__)(pdgid.astype(float))
    X[:, :, 7] = np.vectorize(d_encoding['L1PuppiCands_charge'].__getitem__)(charge.astype(float))
    X[:, :, 8] = np.vectorize(d_encoding['L1PuppiCands_pdgId_org'].__getitem__)(pdgid.astype(float))

    # Gen MET info
    Y = np.zeros(shape=(maxEntries, 4), dtype=float, order='F')
    if not is_data:
        Y[:, 0] += tree['genMet_pt'].to_numpy() * np.cos(tree['genMet_phi'].to_numpy()) # gen MET px
        Y[:, 1] += tree['genMet_pt'].to_numpy() * np.sin(tree['genMet_phi'].to_numpy()) # gen MET py
        Y[:, 2] += tree['genMet_pt'].to_numpy()
        Y[:, 3] += tree['genMet_phi'].to_numpy()

    # L1T MET info
    Z = np.zeros(shape=(maxEntries, 4), dtype=float, order='F')
    Z[:, 0] += tree[l1met_list[0]].to_numpy() * np.cos(tree[l1met_list[1]].to_numpy()) # L1T MET px
    Z[:, 1] += tree[l1met_list[0]].to_numpy() * np.sin(tree[l1met_list[1]].to_numpy()) # L1T MET py
    Z[:, 2] += tree[l1met_list[0]].to_numpy()
    Z[:, 3] += tree[l1met_list[1]].to_numpy()

    # # CMSSW DeepMET info
    # REF = np.zeros(shape=(maxEntries, 4), dtype=float, order='F')
    # REF[:, 0] += tree[cmssw_ml_list[0]].to_numpy() * np.cos(tree[cmssw_ml_list[1]].to_numpy()) # CMSSW DeepMET px
    # REF[:, 1] += tree[cmssw_ml_list[0]].to_numpy() * np.sin(tree[cmssw_ml_list[1]].to_numpy()) # CMSSW DeepMET py
    # REF[:, 2] += tree[cmssw_ml_list[0]].to_numpy()
    # REF[:, 3] += tree[cmssw_ml_list[1]].to_numpy()

    if "15_1_X" in input_path:
        tag = "_151X"
    elif "14_2_X" in input_path:
        tag = "_142X"
    else:
        tag = ""

    outname = f"{evt_name}_PU200_{maxEntries//1000}k{tag}.h5"

    with h5py.File(outname, 'w') as h5f:
        h5f.create_dataset('X',    data=X,   compression='lzf')
        h5f.create_dataset('Y',    data=Y,   compression='lzf')
        h5f.create_dataset('Z',    data=Z,   compression='lzf')
        # h5f.create_dataset('REF',    data=REF,   compression='lzf')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file",)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for sample in cfg["samples"]:
        evt_name = sample.get("name", "unknown")
        input_path = sample["input"]
        maxevents = sample.get("maxevents", -1)
        is_data = sample.get("data", False)

        print(f"\nProcessing sample: {evt_name}")
        convertNanoToHDF5(
            input_path=input_path,
            evt_name=evt_name,
            maxevents=maxevents,
            is_data=is_data,
        )

if __name__ == "__main__":

    main()