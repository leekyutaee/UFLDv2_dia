import torch  # PyTorch 라이브러리
from PIL import Image  # 이미지 처리 라이브러리
import os  # 파일 및 디렉토리 관리 라이브러리
import pdb  # 디버깅을 위한 라이브러리
import numpy as np  # 수치 계산 라이브러리
import cv2  # OpenCV 이미지 처리 라이브러리
from data.mytransforms import find_start_pos  # 차선 시작 위치를 찾는 함수


# 이미지 로딩 함수 (PIL 사용)
def loader_func(path):
    return Image.open(path)


# 테스트 데이터셋 클래스 정의
class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, crop_size=None):
        super(LaneTestDataset, self).__init__()
        self.path = path  # 데이터 경로
        self.img_transform = img_transform  # 이미지 변환 함수
        self.crop_size = crop_size  # 이미지 크롭 크기

        # 이미지 리스트 파일을 읽어서 저장
        with open(list_path, 'r') as f:
            self.list = f.readlines()

        # CULane 데이터셋의 잘못된 경로 처리 ("/" 제거)
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  

    def __getitem__(self, index):
        name = self.list[index].split()[0]  # 이미지 파일명 가져오기
        img_path = os.path.join(self.path, name)  # 전체 이미지 경로 생성
        img = loader_func(img_path)  # 이미지 로드

        if self.img_transform is not None:
            img = self.img_transform(img)  # 변환 적용
        img = img[:, -self.crop_size:, :]  # 이미지 하단 부분 크롭

        return img, name  # 변환된 이미지와 파일명 반환

    def __len__(self):
        return len(self.list)  # 데이터셋 크기 반환


# 학습 데이터셋 클래스 정의
class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, target_transform=None, simu_transform=None,
                 griding_num=50, load_name=False, row_anchor=None, use_aux=False, segment_transform=None, num_lanes=4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform  # 이미지 변환 함수
        self.target_transform = target_transform  # 타겟 변환 함수
        self.segment_transform = segment_transform  # 세그멘테이션 변환 함수
        self.simu_transform = simu_transform  # 시뮬레이션 변환 함수
        self.path = path  # 데이터 경로
        self.griding_num = griding_num  # 그리드 개수
        self.load_name = load_name  # 파일명 로딩 여부
        self.use_aux = use_aux  # 추가 정보 사용 여부
        self.num_lanes = num_lanes  # 차선 개수

        # 이미지 리스트 파일을 읽어서 저장
        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor  # 차선 위치 앵커 설정
        self.row_anchor.sort()  # 차선 위치 정렬

    def __getitem__(self, index):
        l = self.list[index]  # 현재 샘플 데이터 가져오기
        l_info = l.split()  # 파일명과 라벨 파일명 분리
        img_name, label_name = l_info[0], l_info[1]
        
        # CULane 데이터셋 경로 수정
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)  # 라벨 이미지 경로 생성
        label = loader_func(label_path)  # 라벨 이미지 로드

        img_path = os.path.join(self.path, img_name)  # 이미지 경로 생성
        img = loader_func(img_path)  # 이미지 로드

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)  # 시뮬레이션 변환 적용

        lane_pts = self._get_index(label)  # 라벨에서 차선 좌표 추출

        w, h = img.size  # 이미지 크기 가져오기
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)  # 차선 좌표를 분류 레이블로 변환

        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)  # 세그멘테이션 변환 적용

        if self.img_transform is not None:
            img = self.img_transform(img)  # 이미지 변환 적용

        if self.use_aux:
            return img, cls_label, seg_label  # 추가 정보를 포함한 반환
        if self.load_name:
            return img, cls_label, img_name  # 파일명을 포함한 반환
        return img, cls_label  # 기본 반환

    def __len__(self):
        return len(self.list)  # 데이터셋 크기 반환

    # 차선 좌표를 분류 레이블로 변환하는 함수
    def _grid_pts(self, pts, num_cols, w):
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)  # 그리드 샘플링

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]  # y 좌표 가져오기
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    # 라벨에서 차선 좌표를 추출하는 함수
    def _get_index(self, label):
        w, h = label.size

        if h != 288:
            scale_f = lambda x: int((x * 1.0 / 288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))  # 차선 좌표 초기화
        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]  # 특정 차선 픽셀 찾기
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1  # 차선이 없으면 -1 설정
                    continue
                pos = np.mean(pos)  # 차선 중심 위치 계산
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # 데이터 보정: 이미지 경계까지 차선을 확장
        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue  # 유효한 차선이 없으면 스킵

            valid = all_idx_cp[i, :, 1] != -1  # 유효한 차선 점 찾기
            valid_idx = all_idx_cp[i, valid, :]  # 유효한 점만 추출

            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                continue  # 차선이 이미 이미지 바닥까지 닿으면 스킵
            if len(valid_idx) < 6:
                continue  # 유효한 점이 너무 적으면 스킵

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)  # 1차 다항 회귀 적용
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted  # 확장된 차선 적용

        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()  # 디버깅용 중단점 설정

        return all_idx_cp  # 최종 차선 좌표 반환
