import os
import sys
import yaml
import uproot
import argparse
import importlib
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# print("Built with CUDA:", tf.test.is_built_with_cuda())
# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# tf.config.set_visible_devices(gpus[0], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True) 


def make_perfNano_rootfile_from_h5(h5_path, output_path, model, model_type, utils, norm_fac=1.0):
    print(f"Reading h5: {h5_path}")

    if model_type != "TinyTables":
        raise NotImplementedError("Currently only TinyTables is implemented for h5 input.")

    # Read h5 using training snapshot utils
    Xorg, Y, Z = utils.read_input([h5_path])

    # Same preprocessing as training
    X1, X2 = utils.preProcessingForTinyTableModel(Xorg, norm_fac)
    Xr = [X1, X2]

    maxEntries = len(Y)
    print(f"Found {maxEntries} events")

    # Gen MET
    gen_px = norm_fac * Y[:, 0]
    gen_py = norm_fac * Y[:, 1]
    gen_pt = Y[:, 2]
    gen_phi = Y[:, 3]

    # L1/PUPPI MET
    l1_px = norm_fac * Z[:, 0]
    l1_py = norm_fac * Z[:, 1]
    l1_pt = Z[:, 2]
    l1_phi = Z[:, 3]

    # ML MET
    print("Running model prediction...")
    outMet = model.predict(Xr, verbose=0)

    ml_px = norm_fac * outMet[:, 0]
    ml_py = norm_fac * outMet[:, 1]
    ml_pt = np.hypot(ml_px, ml_py)
    ml_phi = np.arctan2(ml_py, ml_px)

    # Output path
    out_root_path = output_path
    if not out_root_path.endswith(".root"):
        out_root_path += ".root"

    out_dir = os.path.dirname(out_root_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write ROOT
    with uproot.recreate(out_root_path) as f:
        f.mktree(
            "Events",
            {
                "l1tMETMet_pt":    "float32",
                "l1tMETMet_phi":   "float32",
                "l1tMETMLMet_pt":  "float32",
                "l1tMETMLMet_phi": "float32",
                "genMet_pt":       "float32",
                "genMet_phi":      "float32",
            },
        )

        f["Events"].extend(
            {
                "l1tMETMet_pt":    l1_pt.astype(np.float32),
                "l1tMETMet_phi":   l1_phi.astype(np.float32),
                "l1tMETMLMet_pt":  ml_pt.astype(np.float32),
                "l1tMETMLMet_phi": ml_phi.astype(np.float32),
                "genMet_pt":       gen_pt.astype(np.float32),
                "genMet_phi":      gen_phi.astype(np.float32),
            }
        )

    print(f"[OK] wrote {maxEntries} events to: {out_root_path}")

    return out_root_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config file"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_dir = cfg["run"]["dir"]
    code_snapshot_dir = os.path.join(run_dir, "code_snapshot")

    if not os.path.isdir(code_snapshot_dir):
        raise RuntimeError(f"code_snapshot directory not found: {code_snapshot_dir}")

    sys.path.insert(0, code_snapshot_dir)

    utils = importlib.import_module("utils")
    models = importlib.import_module("models")
    loss = importlib.import_module("loss")

    globals()["to_np_array"] = utils.to_np_array
    # globals()["preProcessingForTinyTableModel"] = utils.preProcessingForTinyTableModel
    # globals()["preProcessing"] = utils.preProcessing

    model_path = os.path.join(run_dir, cfg["model"]["path"])
    model_type = cfg["model"]["type"]

    output_dir = os.path.join(run_dir, cfg["output"]["dir"])
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using code snapshot: {code_snapshot_dir}")
    print(f"Loading model: {model_path}")

    # model = tf.keras.models.load_model(model_path, compile=False)
    custom_objects = {}

    if hasattr(models, "QuantizeWeightSTE"):
        custom_objects["QuantizeWeightSTE"] = models.QuantizeWeightSTE
    if hasattr(models, "FixedPointClipConstraint"):
        custom_objects["FixedPointClipConstraint"] = models.FixedPointClipConstraint
    if hasattr(models, "SumParticles"):
        custom_objects["SumParticles"] = models.SumParticles
    if hasattr(models, "FeatureSelector"):
        custom_objects["FeatureSelector"] = models.FeatureSelector

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )

    for sample in cfg["samples"]:
        sample_name = sample["name"]
        input_path = sample["input"]
        output_path = os.path.join(output_dir, sample["output"])

        print("=" * 80)
        print(f"Processing sample: {sample_name}")
        print(f"Input : {input_path}")
        print(f"Output: {output_path}")
        print("=" * 80)

        make_perfNano_rootfile_from_h5(
            h5_path=input_path,
            output_path=output_path,
            model=model,
            model_type=model_type,
            utils=utils,
            norm_fac=cfg["model"].get("norm_factor", 1.0),
        )


if __name__ == "__main__":
    main()