# L1METML

### Setup
```bash
git clone git@github.com:jmduarte/L1METML.git
cd L1METML
```

Create an anaconda environment with Python 3.6 and install packages needed to run train.py.
```bash
bash conda_setup.sh
```

### Convert
The TTBar sample used in `convert-uproot.py` is located in
```
/afs/cern.ch/work/d/daekwon/public/L1PF_110X/CMSSW_11_1_2/src/FastPUPPI/NtupleProducer/python/TTbar_PU200_110X_1M/
```
or at this CERNBox link: https://cernbox.cern.ch/index.php/s/JK2InUjatHFxFbf

Convert into HDF5
```
python convertNanoToHDF5_L1triggerToDeepMET.py -i [input .root file path] -o [output file path]
```

### Train
```bash
python train.py --workflowType ['dataGenerator' or 'loadAllData': either use a data generator or load all data into memory]  --input [path to input files] --output [output path (plot and weight will be stored)] --mode [0 or 1 (0 for L1MET model, 1 for DeepMET model)] --epochs [int] --quantized [total bits] [int bits] --units [dense layer 1 units] [dense layer 2 units] [etc.]
```
For example,
```bash
python train.py --workflowType dataGenerator --input ./path/to/files/ --output ./path/to/result/ --mode 1 --epochs --quantized 8 2 --units 12 36
```

### Test
You need output results with input TTbar and SingleNeutrino.
When you use SingleNeutrino sample as input please change 'TTbar' to 'SingleNeutrino' in test function in train.py. #L66-67

SingleNeutrino sample is located in : https://cernbox.cern.ch/index.php/s/5inLVZpXreq1vOx

```bash
python rate_test.py --input [path to input files (output path of train.py)] --plot [ROC, rate, rate_com]
```


# L1METML

### Setup
```bash
git clone git@github.com:jmduarte/L1METML.git
cd L1METML
```

Create an anaconda environment with Python 3.6 and install packages needed to run train.py.
```bash
bash conda_setup.sh
```

### Convert
The TTBar sample used in `convert-uproot.py` is located in
```
/afs/cern.ch/work/d/daekwon/public/L1PF_110X/CMSSW_11_1_2/src/FastPUPPI/NtupleProducer/python/TTbar_PU200_110X_1M/
```
or at this CERNBox link: https://cernbox.cern.ch/index.php/s/JK2InUjatHFxFbf

Convert into HDF5
```
python convertNanoToHDF5.py -i [input .root file path] -o [output file path]

140X
python3 convertNanoToHDF5.py -i ../../dataset/root/140X/perfNano_QCD_Pt15To3000_PU200_L1METML_95k.root -o ../../dataset/h5_input/140X/QCD_Pt15To3000
python3 convertNanoToHDF5.py -i ../../dataset/root/140X/perfNano_SingleNeutrino_PU200_L1METML_300k.root -o ../../dataset/h5_input/140X/SingleNeutrino
python3 convertNanoToHDF5.py -i ../../dataset/root/140X/perfNano_TT_PU200_L1METML_500k.root -o ../../dataset/h5_input/140X/TTbar
python3 convertNanoToHDF5.py -i ../../dataset/root/140X/perfNano_TTTo2L2Nu_PU200_L1METML_500k.root -o ../../dataset/h5_input/140X/TTTo2L2Nu
python3 convertNanoToHDF5.py -i ../../dataset/root/140X/perfNano_TTToSemileptonic_PU200_L1METML_300k.root -o ../../dataset/h5_input/140X/TTToSemileptonic
python3 convertNanoToHDF5.py -i ../../dataset/root/140X/perfNano_VBFHToInvisible_PU200_L1METML_300k.root -o ../../dataset/h5_input/140X/VBFHToInvisible
python3 convertNanoToHDF5.py -i ../../dataset/root/140X/perfNano_WJetsToLNu_PU200_32663.root -o ../../dataset/h5_input/140X/WJetsToLNu

151X
python3 convertNanoToHDF5.py -i ../../dataset/root/151X/perfNano_QCD_15_3000_151X_96k.root -o ../../dataset/h5_input/151X/QCD_15_3000
python3 convertNanoToHDF5.py -i ../../dataset/root/151X/perfNano_TTbar_300k_151X.root -o ../../dataset/h5_input/151X/TTbar
python3 convertNanoToHDF5.py -i ../../dataset/root/151X/perfNano_TTTo2L2Nu_PU200_500k_151X.root -o ../../dataset/h5_input/151X/TTTo2L2Nu
python3 convertNanoToHDF5.py -i ../../dataset/root/151X/perfNano_VBF_HToInvisible_151X.root -o ../../dataset/h5_input/151X/VBF_HToInvisible
python3 convertNanoToHDF5.py -i ../../dataset/root/151X/perfNano_WJetsToLNu_PU200_34072_151X.root -o ../../dataset/h5_input/151X/WJetsToLNu
python3 convertNanoToHDF5.py -i ../../dataset/root/151X/perfNano_TTToSemileptonic_Full.root -o ../../dataset/h5_input/151X/TTToSemileptonic
```

### Train
```bash
python train.py --workflowType ['dataGenerator' or 'loadAllData': either use a data generator or load all data into memory]  --input [path to input files] --output [output path (plot and weight will be stored)] --mode [0 or 1 (0 for L1MET model, 1 for DeepMET model)] --epochs [int] --quantized [total bits] [int bits] --units [dense layer 1 units] [dense layer 2 units] [etc.]
```
For example,
```bash
python3 train.py --config config/train.yml