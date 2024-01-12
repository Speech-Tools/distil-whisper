"""
wav와 txt로 이루어진 원본 데이터셋을
huggingface dataset 객체로 구성해 distil-whisper에 사용할 수 있게 만드는 코드

입력으로 받는 `db_dir` 안에는 기본적으로 훈련용 데이터셋 디렉터리인 train_data_01과
테스트용 데이터셋 디렉터리인 test_data_01이 존재해야 한다.

Usage:
    python data_prep.py \
        --db_dir=<데이터셋 루트 디렉터리> \
        --out_dir=<데이터가 저장될 디렉터리>

e.g.:
    python data_prep.py \
        --db_dir=speechDATA \
        --out_dir=data

"""
import math
import argparse
import subprocess
import numpy as np
import librosa
import csv
import pyarrow as pa
from datasets import Dataset
from pathlib import Path
from typing import Union, List, Tuple
from multiprocessing import Pool


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distil-whisper-ko data preparation")

    parser.add_argument(
        "--db_dir",
        type=Path,
        required=True,
        help="Database Root directory."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="A directory path to save arrow data files."
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=16,
        help="A number of multiprocess.",
    )
    return parser


def process_one(txt_file: Tuple[int, Path]):
    i, txt_file_path = txt_file
        
    wav_file = str(txt_file_path).replace('label', 'source')
    wav_file = Path(wav_file).with_suffix('.wav')
    
    result = {
        'path': None,
        'audio': None,
        'sentence': None,
    }
    
    if not wav_file.is_file():
        # wav 파일이 없다면 이번 데이터는 패스
        print(f"No matching wav file to '{txt_file_path}'")
        return result

    # Prepare text
    with open(txt_file_path, 'r') as f:
        utt = f.read().strip()
        utt = utt.replace('\n',' ')

        # utt가 없다면 패스
        if len(utt) < 1:
            return result

    # 필요한 것은 오디오 데이터 경로, 오디오 데이터, 오디오 데이터 샘플링 레이트, 텍스트 데이터
    wav_data, sr = librosa.load(wav_file, sr=16000)
    
    result = {
        'path': str(wav_file),
        'audio': {
            'path': str(wav_file),
            'array': pa.array(wav_data),
            'sampling_rate': sr,
        },
        'sentence': utt,
    }
    return result


def work(txt_files: List[Tuple[int, Path]]):
    outputs = list(map(process_one, txt_files))
    outputs = [output for output in outputs if output['path'] != None]
    print(f"Done multiprocess id: {txt_files[0][0]}")
    return outputs


def prepare_data(
    db_dir: Union[str, Path],
    out_dir: Union[str, Path],
    nj: int = 256,
):
    db_dir = Path(db_dir)
    out_dir = Path(out_dir)
    
    src_dir = db_dir.absolute()
    dst_dir = out_dir.absolute()

    if not dst_dir.is_dir():
        dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        raise FileNotFoundError(f"No such directorry: {src_dir}")

    # Prepare `text`
    ## '.txt' 파일 탐색
    ## utt id는 순서대로 임의로 붙이기
    print(f"Start find txt files in {src_dir}")

    target_txts = list(enumerate(sorted(src_dir.rglob('*.txt'))))
    print(f"Length of dataset: {len(target_txts)}")
    target_txts = [(str(target_txt[0]).zfill(10), target_txt[1]) for target_txt in target_txts]
    div = math.ceil(len(target_txts) / nj)
    div_target_txts = [target_txts[i*div:(i+1)*div] for i in range((len(target_txts)+div-1) // div)]

    with Pool(nj) as p:
        results = p.map(work, div_target_txts)

    # Parsing 후 저장
    paths = []
    wavs = []
    texts = []
    for result in results:
        for temp_data in result:
            paths.append(temp_data['path'])
            wavs.append(temp_data['audio'])
            texts.append(temp_data['sentence'])

    final_result = {
        'path': paths,
        'audio': wavs,
        'sentence': texts,
    }
    
    final_dataset = Dataset.from_dict(final_result)
    final_dataset.save_to_disk(dst_dir)
    print(f"Save dataset to {dst_dir}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    # data_parts = ["test_data_01", "train_data_01"]
    data_parts = ["train_data_01"]
    for data_part in data_parts:
        print(f"Preparing {args.db_dir / data_part}")
        prepare_data(args.db_dir / data_part / "고객응대", args.out_dir / data_part, nj=args.nj)

    print("Done data_prep")


if __name__=="__main__":
    main()
