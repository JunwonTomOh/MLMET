#!/usr/bin/env python

import optparse
import sys
import uproot
import numpy as np
import awkward as ak
import h5py
#import progressbar
from tqdm import tqdm
import os

from utils import to_np_array

'''
widgets=[
    progressbar.SimpleProgress(), ' - ', progressbar.Timer(), ' - ', progressbar.Bar(), ' - ', progressbar.AbsoluteETA()
]
'''


def deltaR(eta1, phi1, eta2, phi2):
    """ calculate deltaR """
    dphi = (phi1-phi2)
    while dphi > np.pi:
        dphi -= 2*np.pi
    while dphi < -np.pi:
        dphi += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)


# configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input', help='input file', default='', type='string')
parser.add_option('-o', '--output', dest='output', help='output file', default='', type='string')
parser.add_option("-N", "--maxevents", dest='maxevents', help='max number of events', default=-1, type='int')
parser.add_option("--data", dest="data", action="store_true", default=False, help="input is data. The default is MC")
(opt, args) = parser.parse_args()

if opt.input == '' or opt.output == '':
    sys.exit('Need to specify input and output files!')

##
varList = [
        'nL1PuppiCands', 'L1PuppiCands_pt', 'L1PuppiCands_eta', 'L1PuppiCands_phi',
        'L1PuppiCands_charge', 'L1PuppiCands_pdgId', 'L1PuppiCands_puppiWeight'
]

# event-level variables

varList_mc = [
    'genMet_pt', 'genMet_phi',
]
tag = "_151X" if "151X" in opt.input else ""

if "151X" in opt.input:
    l1met_list = ['ctl2METMet_pt', 'ctl2METMet_phi']
else:
    l1met_list = ['l1tMETMet_pt', 'l1tMETMet_phi']


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


if not opt.data:
    varList = varList + varList_mc +l1met_list

upfile = uproot.open(opt.input)

if opt.maxevents > 0:
    tree = upfile["Events"].arrays(varList, entry_stop=opt.maxevents)
else:
    tree = upfile["Events"].arrays(varList)

# general setup
maxNPuppi = 128
nFeatures = 9
maxEntries = len(tree["nL1PuppiCands"])

print(f"Found {maxEntries} Events")

# input Puppi candidates
X = np.zeros(shape=(maxEntries, maxNPuppi, nFeatures), dtype=float, order='F')
# recoil estimators
Y = np.zeros(shape=(maxEntries, 4), dtype=float, order='F')
# L1T MET
Z = np.zeros(shape=(maxEntries, 4), dtype=float, order='F')

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

# truth info
if not opt.data:
    Y[:, 0] += tree['genMet_pt'].to_numpy() * np.cos(tree['genMet_phi'].to_numpy()) # gen MET px
    Y[:, 1] += tree['genMet_pt'].to_numpy() * np.sin(tree['genMet_phi'].to_numpy()) # gen MET py
    Y[:, 2] += tree['genMet_pt'].to_numpy()
    Y[:, 3] += tree['genMet_phi'].to_numpy()

# L1T MET info
Z[:, 0] += tree[l1met_list[0]].to_numpy() * np.cos(tree[l1met_list[1]].to_numpy()) # L1T MET px
Z[:, 1] += tree[l1met_list[0]].to_numpy() * np.sin(tree[l1met_list[1]].to_numpy()) # L1T MET py
Z[:, 2] += tree[l1met_list[0]].to_numpy()
Z[:, 3] += tree[l1met_list[1]].to_numpy()

tag = "_151X" if "151X" in opt.input else ""
outname = f"{opt.output}_PU200_{maxEntries//1000}k{tag}.h5"

with h5py.File(outname, 'w') as h5f:
    h5f.create_dataset('X',    data=X,   compression='lzf')
    h5f.create_dataset('Y',    data=Y,   compression='lzf')
    h5f.create_dataset('Z',    data=Z,   compression='lzf')