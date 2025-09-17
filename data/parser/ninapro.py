# -*- coding:utf-8 -*-
import os
import glob
import polars as pl
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, decimate


def preprocessing(df):
    fs = 2000
    low_cut = 20
    high_cut = 500
    order = 4
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # 1. 필터링 (Filtering)
    df_filtered = df.select(
        [
            # ✅ 이 부분에 return_dtype=pl.Float64 추가
            pl.col(c).map_batches(
                lambda s: filtfilt(b, a, s.to_numpy()),
                return_dtype=pl.Float64
            )
            for c in df.columns
        ]
    )

    # 2. 정류 (Rectification)
    df_rectified = df_filtered.select(pl.all().abs())

    return df_rectified


def parse_ninapro_db2(file_path):
    try:
        mat_data = sio.loadmat(file_path)
        emg_signal = mat_data['emg']
        # print(mat_data)
        # exit()

        # 동작(Exercise) 레이블: (샘플 수 x 1)
        # 0: 휴식, 1~17: 동작 1~17 (Exercise B)
        # 18~40: 동작 1~23 (Exercise C)
        exercise_label = mat_data['stimulus']

        # 반복(Repetition) 레이블: (샘플 수 x 1)
        # 각 동작의 반복 횟수를 나타냄
        repetition_label = mat_data['repetition']

        # 데이터 정보 출력
        print("\n--- 데이터 구조 정보 ---")
        print(f"EMG 신호 형태 (samples, channels): {emg_signal.shape}")
        print(f"동작 레이블 형태 (samples, 1): {exercise_label.shape}")
        print(f"반복 레이블 형태 (samples, 1): {repetition_label.shape}")

        # 고유 레이블 확인
        unique_exercises = np.unique(exercise_label)
        print(f"\n포함된 동작 레이블 (0=휴식): \n{unique_exercises}")

        parsed_data = {
            'emg': emg_signal,
            'exercise': exercise_label,
            'repetition': repetition_label
        }

        return parsed_data

    except FileNotFoundError:
        print(f"🚨 에러: 파일을 찾을 수 없습니다. 경로를 확인하세요: {file_path}")
        return None
    except KeyError as e:
        print(f"🚨 에러: .mat 파일에서 예상되는 변수('{e}')를 찾을 수 없습니다.")
        return None
    except MemoryError as e:
        return None


def preprocessing_combined(df_emg: pl.DataFrame, df_mask: pl.DataFrame):
    # 필터 설계
    fs = 2000
    low_cut = 20
    high_cut = 500
    order = 4
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # 필터링
    df_filtered = df_emg.select(
        [
            pl.col(c).map_batches(
                lambda s: filtfilt(b, a, s.to_numpy()),
                return_dtype=pl.Float64
            )
            for c in df_emg.columns
        ]
    )

    q = 2
    df_downsampled = df_filtered.select(
        [
            pl.col(c).map_batches(
                lambda s: decimate(s.to_numpy(), q),
                return_dtype=pl.Float64
            )
            for c in df_filtered.columns
        ]
    )

    df_emg_processed = df_downsampled.select(pl.all().abs())
    df_mask_processed = df_mask[::q]
    return df_emg_processed, df_mask_processed

def main():
    base_path = os.path.join('/', 'data', 'segmentation', 'ninapro')
    trg_path = os.path.join('/', 'data', 'segmentation', 'ninapro_f2')
    for path in os.listdir(base_path):
        subject_path = os.path.join(base_path, path)
        for name in os.listdir(subject_path):
            mat_path = os.path.join(subject_path, name)

            df = parse_ninapro_db2(mat_path)
            if df is None:
                continue

            data_df = pl.DataFrame(df['emg'])
            # data_df = preprocessing(data_df)

            columns = data_df.columns
            data_df.columns = [f'emg_{i + 1}' for i, _ in enumerate(columns)]
            mask_df = pl.DataFrame(df['exercise'])
            mask_df.columns = ['MASK']

            data_df, mask_df = preprocessing_combined(data_df, mask_df)

            if len(data_df) != len(mask_df):
                continue

            e_path = os.path.join(trg_path, name.split('.')[0].lower())

            if not os.path.exists(e_path):
                os.makedirs(e_path)

            data_df.write_parquet(os.path.join(e_path, 'data.parquet'))
            mask_df.write_parquet(os.path.join(e_path, 'mask.parquet'))


if __name__ == '__main__':
    main()
