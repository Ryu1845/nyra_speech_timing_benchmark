"""Long-running MWA worker, executed inside the MWA environment.

Loads the MWA models once, then reads one JSON request per line on stdin
(``{"audio": <16 kHz mono wav>, "words": [...]}``) and writes one JSON response
per line on stdout (``{"words": [[word, start, end], ...]}`` or
``{"error": "..."}``). Usage::

    python mwa_worker.py <mwa_repo> <model_name> <device>
"""

import json
import os
import sys
import tempfile
import traceback


def main() -> None:
    repo, model_name, device = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, repo)
    os.chdir(repo)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np

    import inference.configuration.constants as constants
    from inference.configuration.get_models_configuration import get_models_configurations
    from inference.configuration.validate_input import UserInput
    from inference.models.dp_algorithm.extract_features import Features_DP
    from inference.models.mms.mms import load_mms_model
    from inference.models.predict import get_file_prediction
    from inference.models.preprocess import prepare_dataset
    from inference.models.unsupSeg.unsupseg_classifier import load_unsupseg_model
    from inference.models.utils import load_model, prepare_sentence

    work = tempfile.mkdtemp(prefix="mwa_worker_")
    example = os.path.join(repo, "inference", "examples", "english.wav")
    user = UserInput(wav_input=example, transcript_input=example.replace(".wav", ".txt"),
                     language="eng", model_name=model_name, device=device,
                     output_folder=work, no_graph=True, no_csv=True)
    config = get_models_configurations(user_parameters=user)
    model = load_model(**config)
    config["mms_bundle"], config["mms_model"] = load_mms_model(config["device"])
    config["unsupseg_model"] = load_unsupseg_model(
        os.path.join(constants.INFERENCE_PART_DIR, config["unsupseg_ckpt"]), config["device"])
    config["_dp_features_obj"] = Features_DP(config["dp_features"])
    config["_w_floats"] = list(config["w"])

    real_stdout.write(json.dumps({"ready": True}) + "\n")
    real_stdout.flush()

    transcript = os.path.join(work, "utt.txt")
    for line in sys.stdin:
        try:
            req = json.loads(line)
            words = req["words"]
            with open(transcript, "w") as f:
                f.write(" ".join(words))
            config["wav_file"] = req["audio"]
            config["transcript_file"] = transcript
            sentence = prepare_sentence(transcript, language="eng")
            batches, masks = prepare_dataset(**config)
            _, _, _, boundaries = get_file_prediction(model, batches, masks, sentence, **config)
            ends = list(np.array(boundaries) / config["dp_times_in_second"])
            starts = [0.0] + ends[:-1]
            out_words = words if len(words) == len(ends) else sentence.split()
            resp = {"words": [[w, float(s), float(e)]
                              for w, s, e in zip(out_words, starts, ends)]}
        except Exception as exc:
            traceback.print_exc()
            resp = {"error": f"{type(exc).__name__}: {exc}"}
        real_stdout.write(json.dumps(resp) + "\n")
        real_stdout.flush()


if __name__ == "__main__":
    main()
