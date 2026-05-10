# Tiny Table MET

### Setup
```
git clone git@github.com:JunwonTomOh/TinyTableMET.git
cd TinyTableMET
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba env create -f environment.yml

# convert perfNano root files into h5 files
python3 convertNanoToHDF5.py --config config/convertNanoToHDF5.yml

# train
python3 train.py --config config/train.yml

# Generate test root files for plotting
python3 save_as_root.py --config config/save_as_root.yml
```

