import argparse

import six
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor
from pathlib import Path

ROOT = Path(__file__).parent
CONFIG = ROOT / "statystyki" / "config.yml"
PATHS_CSV = ROOT / "statystyki" / "paths4_remapped.csv"


def calculate_features(scan, row_dict={}):
    extractor = featureextractor.RadiomicsFeatureExtractor(str(CONFIG))
    mask = np.ones_like(scan)
    mask[0, 0] = 0
    result = extractor.execute(sitk.GetImageFromArray(scan), sitk.GetImageFromArray(mask), voxelBased=False)
    extractor.enableAllImageTypes()
    names = []
    vals = []
    for k, v in row_dict.items():
        names.append(k)
        vals.append(v)
    for key, val in six.iteritems(result):
        if 'diagn' not in key:
            names.append(key)
            vals.append(val)
    return pd.Series(data=vals, index=names)


def get_parser():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-f', '--from', help='Description for foo argument', required=True, type=int)
    parser.add_argument('-t', '--to', help='Description for bar argument', required=True, type=int)
    parser.add_argument('-p', '--prefix', help='Output directory prefix for the result CSV', default='.', type=Path)
    args = vars(parser.parse_args())
    return args


def main():
    args = get_parser()
    df = pd.read_csv(PATHS_CSV)
    print(len(df))
    result = []
    for idx, row in df.iloc[args['from']:args['to']].iterrows():
        print(idx)
        row_dict = dict(row)
        array = np.load(row["full_path"], allow_pickle=True)
        lime = (array['lime'] + 1) * 127.5
        features = calculate_features(lime, row_dict)
        result.append(features)
        pd.DataFrame(result).to_csv(args['prefix'] / f"{args['from']}-{args['to']}_scaling.csv")


if __name__ == "__main__":
    main()
