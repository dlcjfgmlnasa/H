# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data/segmentation/ninapro_f2')
    return parser.parse_args()


def main(args):
    data_path = args.data_path
    for path in os.listdir(data_path)[4:]:
        path = os.path.join(data_path, path)
        print(path)
        data_df = pl.read_parquet(os.path.join(path, 'data.parquet'))
        mask_df = pl.read_parquet(os.path.join(path, 'mask.parquet'))

        data = data_df['emg_2'].to_numpy()
        mask = mask_df.to_numpy().flatten()  # 1차원 배열로 변환

        i = 5
        data = data[10000*i:10000*(i+1)]
        mask = mask[10000*i:10000*(i+1)]
        # 마스크를 이진(0과 1)으로 변환
        mask[mask > 0] = 1

        # --- 시각화 (수정된 부분) ---
        # 1. 그래프 객체 생성
        fig, ax = plt.subplots(figsize=(15, 5))

        # 2. EMG 신호 플로팅
        ax.plot(data, label='EMG Signal', color='blue', linewidth=1)

        # 3. 마스크 값이 1인 구간의 시작/끝점 찾기
        # np.diff를 사용하여 값이 바뀌는 지점을 찾음 (0->1 또는 1->0)
        change_points = np.diff(np.concatenate(([0], mask, [0])))
        start_points = np.where(change_points == 1)[0]
        end_points = np.where(change_points == -1)[0]

        # 4. 각 구간에 대해 배경색 칠하기 (axvspan)
        for start, end in zip(start_points, end_points):
            # axvspan(xmin, xmax, ...)
            ax.axvspan(start, end, color='yellow', alpha=0.4, label='Motion' if start == start_points[0] else "")

        # 5. 그래프 스타일 설정
        ax.set_title('EMG Signal with Motion Segmentation Mask')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()
        exit()

    pass


if __name__ == '__main__':
    main(get_args())
