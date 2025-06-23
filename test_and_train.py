# Copyright 2025 The Neurojourney Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


class Model:
    bpm_mean: float
    bpm_std: float
    sdnn_mean: float
    sdnn_std: float
    hf_p_mean: float
    hf_p_std: float
    ln_hf_mean: float
    ln_hf_std: float
    stress_index_mean: float
    stress_index_std: float
    rsp_stability_mean: float
    rsp_stability_std: float
    cognitive_load_mean: float
    cognitive_load_std: float

    def __init__(self, bpm_mean: float, bpm_std: float,
                 sdnn_mean: float, sdnn_std: float,
                 hf_p_mean: float, hf_p_std: float,
                 ln_hf_mean: float, ln_hf_std: float,
                 stress_index_mean: float, stress_index_std: float,
                 rsp_stability_mean: float, rsp_stability_std: float,
                 cognitive_load_mean: float, cognitive_load_std: float):
        self.bpm_mean = bpm_mean
        self.bpm_std = bpm_std
        self.sdnn_mean = sdnn_mean
        self.sdnn_std = sdnn_std
        self.hf_p_mean = hf_p_mean
        self.hf_p_std = hf_p_std
        self.ln_hf_mean = ln_hf_mean
        self.ln_hf_std = ln_hf_std
        self.stress_index_mean = stress_index_mean
        self.stress_index_std = stress_index_std
        self.rsp_stability_mean = rsp_stability_mean
        self.rsp_stability_std = rsp_stability_std
        self.cognitive_load_mean = cognitive_load_mean
        self.cognitive_load_std = cognitive_load_std


class FocusRegressor:
    focus_model = Model(
        79.32479891, 10.07612672,
        38.55314307, 3.918255556,
        52.6491029, 8.385048483,
        6.155925687, 0.213317513,
        116.9109742, 53.04789398,
        3.736595417, 2.094521305,
        53.40231667, 10.95185607,
    )
    nonfocus_model = Model(
        76.50980237, 8.738497276,
        40.60178858, 3.837145674,
        56.63072725, 7.293126749,
        6.22460578, 0.181063429,
        82.03224006, 31.83219743,
        5.171896955, 2.346611575,
        47.71456394, 8.151117238,
    )

    def load_dataset(self, filename: str):
        f = open(filename, 'r')
        dataset = []
        for line in f.readlines()[1:]:
            split_line = line.split(",")
            data = [
                split_line[0],  # name
                split_line[1],  # task
                int(split_line[2]),  # num
                float(split_line[3]),  # bpm
                float(split_line[4]),  # sdnn
                float(split_line[12]),  # hf_p
                float(split_line[15]),  # ln_hf
                float(split_line[25]),  # stress_index
                float(split_line[29]),  # rsp_stability
                float(split_line[31]),  # cognitive_load
                float(split_line[-1]),  # label
            ]
            dataset.append(data)

        return dataset

    def normalize(self, data, mean, std):
        norm_data = abs(data - mean) / std
        return norm_data

    def normalize_dataset(self, dataset):
        norm_dataset = []
        for data in dataset:
            norm_data = [
                data[0],  # name
                data[1],  # task
                data[2],  # num
                self.normalize(data[3], self.focus_model.bpm_mean, self.focus_model.bpm_std),  # bpm
                self.normalize(data[4], self.focus_model.sdnn_mean, self.focus_model.sdnn_std),  # sdnn
                self.normalize(data[5], self.focus_model.hf_p_mean, self.focus_model.hf_p_std),  # hf_p
                self.normalize(data[6], self.focus_model.ln_hf_mean, self.focus_model.ln_hf_std),  # ln_hf
                self.normalize(data[7], self.focus_model.stress_index_mean, self.focus_model.stress_index_std),  # stress_index
                self.normalize(data[8], self.focus_model.rsp_stability_mean, self.focus_model.rsp_stability_std),  # rsp_stability
                self.normalize(data[9], self.focus_model.cognitive_load_mean, self.focus_model.cognitive_load_std),  # cognitive_load
                self.normalize(data[3], self.nonfocus_model.bpm_mean, self.nonfocus_model.bpm_std),  # bpm
                self.normalize(data[4], self.nonfocus_model.sdnn_mean, self.nonfocus_model.sdnn_std),  # sdnn
                self.normalize(data[5], self.nonfocus_model.hf_p_mean, self.nonfocus_model.hf_p_std),  # hf_p
                self.normalize(data[6], self.nonfocus_model.ln_hf_mean, self.nonfocus_model.ln_hf_std),  # ln_hf
                self.normalize(data[7], self.nonfocus_model.stress_index_mean, self.nonfocus_model.stress_index_std),  # stress_index
                self.normalize(data[8], self.nonfocus_model.rsp_stability_mean, self.nonfocus_model.rsp_stability_std),  # rsp_stability
                self.normalize(data[9], self.nonfocus_model.cognitive_load_mean, self.nonfocus_model.cognitive_load_std),  # cognitive_load
                data[10],  # label
            ]
            norm_dataset.append(norm_data)

        return norm_dataset

    def get_feature(self, data):
        focus_bpm = data[3]
        focus_sdnn = data[4]
        focus_hf_p = data[5]
        focus_ln_hf = data[6]
        focus_stress_index = data[7]
        focus_rsp_stability = data[8]
        focus_cognitive_load = data[9]
        nonfocus_bpm = data[10]
        nonfocus_sdnn = data[11]
        nonfocus_hf_p = data[12]
        nonfocus_ln_hf = data[13]
        nonfocus_stress_index = data[14]
        nonfocus_rsp_stability = data[15]
        nonfocus_cognitive_load = data[16]
        focus_avg7 = np.average([focus_bpm,
                                 focus_sdnn,
                                 focus_hf_p,
                                 focus_ln_hf,
                                 focus_stress_index,
                                 focus_rsp_stability,
                                 focus_cognitive_load])
        nonfocus_avg7 = np.average([nonfocus_bpm,
                                    nonfocus_sdnn,
                                    nonfocus_hf_p,
                                    nonfocus_ln_hf,
                                    nonfocus_stress_index,
                                    nonfocus_rsp_stability,
                                    nonfocus_cognitive_load])
        return [focus_avg7, nonfocus_avg7]

    def get_train_dataset(self, dataset):
        X_train = []
        y_train = []
        for data in dataset:
            task = data[1]
            num = data[2]
            # if num == 1:
            feature = self.get_feature(data)
            label = data[-1]
            X_train.append(feature)
            y_train.append(label)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        return X_train, y_train

    def add_gaussian_noise(self, X, mean=0.0, std=0.01):
        noise = np.random.normal(mean, std, X.shape)
        return X + noise

    def augmentation(self, X, y):
        X1 = self.add_gaussian_noise(X, std=0.01)
        X2 = self.add_gaussian_noise(X, std=0.02)
        X3 = self.add_gaussian_noise(X, std=0.03)
        X4 = self.add_gaussian_noise(X, std=0.04)
        X5 = self.add_gaussian_noise(X, std=0.05)
        X6 = self.add_gaussian_noise(X, std=0.001)
        X7 = self.add_gaussian_noise(X, std=0.002)
        X8 = self.add_gaussian_noise(X, std=0.003)
        X9 = self.add_gaussian_noise(X, std=0.004)
        X10 = self.add_gaussian_noise(X, std=0.005)
        X11 = self.add_gaussian_noise(X, std=0.006)
        X12 = self.add_gaussian_noise(X, std=0.007)
        X13 = self.add_gaussian_noise(X, std=0.008)
        X14 = self.add_gaussian_noise(X, std=0.009)

        X_aug = np.vstack([X, X1, X2, X3, X4, X5, X6, X7, X8,
                           X9, X10, X11, X12, X13, X14])
        y_aug = np.concatenate([y, y, y, y, y, y, y, y, y,
                                y, y, y, y, y, y])

        return X_aug, y_aug

    def train(self, X_train, y_train, model_name="focus_model.pkl"):
        from xgboost import XGBRegressor

        # 모델 학습
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)

        # 예측
        y_pred = model.predict(X_train)

        # 평가 지표
        r2 = r2_score(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        rmse = mean_squared_error(y_train, y_pred)
        corr = np.corrcoef(y_train, y_pred)[0, 1]

        # 정확도 계산 (0.5 기준 이진 분류로 처리)
        y_true_binary = (y_train >= 0.5).astype(int)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        acc = accuracy_score(y_true_binary, y_pred_binary)

        # 결과 출력
        print(f"📊 Train Results:")
        print(f"  R² Score      : {r2:.4f}")
        print(f"  MAE           : {mae:.4f}")
        print(f"  RMSE          : {rmse:.4f}")
        print(f"  Correlation   : {corr:.4f}")
        print(f"  Accuracy      : {acc:.4f}")

        # 모델 저장
        joblib.dump(model, model_name)

    def test(self, X_test, y_test, model_name="focus_model.pkl"):
        # 모델 로드
        model = joblib.load(model_name)

        # 예측
        y_pred = model.predict(X_test)

        # 예측값을 0~1 범위로 클리핑 (필요한 경우)
        y_pred = np.clip(y_pred, 0.0, 1.0)

        # 평가 지표 계산
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        corr = np.corrcoef(y_test, y_pred)[0, 1]

        # 정확도 계산 (0.5 기준 이진 분류로 처리)
        y_true_binary = (y_test >= 0.5).astype(int)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        acc = accuracy_score(y_true_binary, y_pred_binary)

        # 결과 출력
        print(f"📊 Test Results:")
        print(f"  R² Score      : {r2:.4f}")
        print(f"  MAE           : {mae:.4f}")
        print(f"  RMSE          : {rmse:.4f}")
        print(f"  Correlation   : {corr:.4f}")
        print(f"  Accuracy      : {acc:.4f}")

        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'correlation': corr,
            'y_pred': y_pred
        }


def main():
    focus_regressor = FocusRegressor()

    # Load dataset
    dataset = focus_regressor.load_dataset("dataset.csv")

    # Normalize
    norm_dataset = focus_regressor.normalize_dataset(dataset)

    # Get train dataset
    X_train, y_train = focus_regressor.get_train_dataset(norm_dataset)

    # Augmentation
    X_train, y_train = focus_regressor.augmentation(X_train, y_train)

    # Get train / test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, shuffle=True
    )

    # Train
    focus_regressor.train(X_train, y_train)

    # Test
    focus_regressor.test(X_test, y_test)


if __name__ == "__main__":
    main()
