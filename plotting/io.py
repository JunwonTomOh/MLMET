import os
import numpy as np
import uproot


DEFAULT_BRANCHES = [
    "l1tMETMet_pt",
    "l1tMETMet_phi",
    "l1tMETMLMet_pt",
    "l1tMETMLMet_phi",
    "genMet_pt",
    "genMet_phi",
]


def read_met_root(root_path, tree_name="Events", branches=None):
    """
    Read MET branches from a flat ROOT file.

    Expected branches:
      - l1tMETMet_pt
      - l1tMETMet_phi
      - l1tMETMLMet_pt
      - l1tMETMLMet_phi
      - genMet_pt
      - genMet_phi

    Returns:
      dict with short names:
        gen_pt, gen_phi
        l1_pt, l1_phi
        ml_pt, ml_phi
    """

    if branches is None:
        branches = DEFAULT_BRANCHES

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    with uproot.open(root_path) as f:
        if tree_name not in f:
            raise KeyError(f"Tree '{tree_name}' not found in {root_path}")

        tree = f[tree_name]

        missing = [b for b in branches if b not in tree.keys()]
        if missing:
            raise KeyError(
                f"Missing branches in {root_path}: {missing}\n"
                f"Available branches: {tree.keys()}"
            )

        arrays = tree.arrays(branches, library="np")

    data = {
        "gen_pt": np.asarray(arrays["genMet_pt"], dtype=np.float32),
        "gen_phi": np.asarray(arrays["genMet_phi"], dtype=np.float32),

        "l1_pt": np.asarray(arrays["l1tMETMet_pt"], dtype=np.float32),
        "l1_phi": np.asarray(arrays["l1tMETMet_phi"], dtype=np.float32),

        "ml_pt": np.asarray(arrays["l1tMETMLMet_pt"], dtype=np.float32),
        "ml_phi": np.asarray(arrays["l1tMETMLMet_phi"], dtype=np.float32),
    }

    return data


def read_many_met_roots(samples, input_dir=None, tree_name="Events"):
    """
    Read multiple ROOT files.

    samples example:
      [
        {"name": "TTToSemileptonic", "file": "TTToSemileptonic_ml.root"},
        {"name": "SingleNeutrino", "file": "SingleNeutrino_ml.root"},
      ]

    Returns:
      dict:
        {
          "TTToSemileptonic": data_dict,
          "SingleNeutrino": data_dict,
        }
    """

    out = {}

    for sample in samples:
        name = sample["name"]
        file_path = sample.get("file", sample.get("path", None))

        if file_path is None:
            raise KeyError(f"Sample '{name}' has no 'file' or 'path' field.")

        if input_dir is not None and not os.path.isabs(file_path):
            root_path = os.path.join(input_dir, file_path)
        else:
            root_path = file_path

        print(f"[io] Reading {name}: {root_path}")
        out[name] = read_met_root(root_path, tree_name=tree_name)

    return out


def get_reco(data, kind):
    """
    Convenience function.

    kind:
      - "l1"
      - "ml"
      - "gen"

    Returns:
      pt, phi
    """

    if kind == "l1":
        return data["l1_pt"], data["l1_phi"]

    if kind == "ml":
        return data["ml_pt"], data["ml_phi"]

    if kind == "gen":
        return data["gen_pt"], data["gen_phi"]

    raise ValueError(f"Unknown kind: {kind}. Choose from 'gen', 'l1', 'ml'.")