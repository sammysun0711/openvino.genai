from pathlib import Path
import nncf
from glm4v_helper import convert_glm4v_model
import argparse

def main():
    
    parser = argparse.ArgumentParser('glm4v ov convert tool', add_help=True, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the model is stored')
    parser.add_argument('--use_int4', action='store_true', help='Compress to int4')
    args = parser.parse_args()
    model_dir = args.model_dir
    use_int4 = args.use_int4
    out_dir = ""
    if use_int4:
        compression_configuration = {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 128,
            "ratio": 1.0,
        }
        out_dir = "glm4v-nano-v050-ov-int4"
    else:
        compression_configuration = None
        out_dir = "glm4v-nano-v050-ov-fp16"

    convert_glm4v_model(model_dir, out_dir, compression_configuration)

if __name__ == '__main__':
    main()
