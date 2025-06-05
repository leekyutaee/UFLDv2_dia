import torch  # PyTorch 라이브러리
from model.backbone import resnet  # ResNet 백본(backbone) 네트워크 임포트
import numpy as np  # 수치 계산 라이브러리
from utils.common import initialize_weights  # 가중치 초기화 함수 임포트
from model.seg_model import SegHead  # 세그멘테이션 헤드(SegHead) 클래스 임포트
from model.layer import CoordConv  # 좌표 컨볼루션(CoordConv) 레이어 임포트

# 차선 검출 네트워크 클래스 정의
class parsingNet(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row=None, num_cls_row=None, num_grid_col=None, num_cls_col=None, 
                 num_lane_on_row=None, num_lane_on_col=None, use_aux=False, input_height=None, input_width=None, fc_norm=False):
        super(parsingNet, self).__init__()

        # 입력 파라미터 저장
        self.num_grid_row = num_grid_row  # 세로 방향 격자(grid) 개수
        self.num_cls_row = num_cls_row  # 세로 방향 분류(classification) 개수
        self.num_grid_col = num_grid_col  # 가로 방향 격자(grid) 개수
        self.num_cls_col = num_cls_col  # 가로 방향 분류(classification) 개수
        self.num_lane_on_row = num_lane_on_row  # 세로 방향 차선 개수 ##############################################
        self.num_lane_on_col = num_lane_on_col  # 가로 방향 차선 개수 ##############################################
        self.use_aux = use_aux  # 보조 네트워크 사용 여부

        # 출력 차원 계산
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row  # 존재 유무 예측을 위한 추가 출력
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col  # 존재 유무 예측을 위한 추가 출력
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4  # 전체 출력 차원 계산

        mlp_mid_dim = 2048  # MLP의 중간 레이어 차원 설정
        self.input_dim = (input_height // 32) * (input_width // 32) * 8  # 입력 이미지 크기 기반으로 특징 추출 후 차원 계산

        # ResNet 백본(backbone) 모델 초기화
        self.model = resnet(backbone, pretrained=pretrained)

        # 평균 풀링 혹은 최대 풀링을 사용한 실험 코드 (비활성화 상태)
        # self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.pool = torch.nn.AdaptiveMaxPool2d(1)

        ####실제 예측 하는곳# 완전 연결층 (FC) 정의
        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),  # 입력 정규화 옵션
            torch.nn.Linear(self.input_dim, mlp_mid_dim),  # 첫 번째 FC 레이어
            torch.nn.ReLU(),  # 활성화 함수
            torch.nn.Linear(mlp_mid_dim, self.total_dim),  # 두 번째 FC 레이어 (최종 출력 크기)
        )

        # 특징 맵을 축소하는 컨볼루션 레이어 정의 (백본 크기에 따라 다르게 설정)
        self.pool = torch.nn.Conv2d(512, 8, 1) if backbone in ['34', '18', '34fca'] else torch.nn.Conv2d(2048, 8, 1)

        # 보조 네트워크 사용 시 세그멘테이션 헤드 추가
        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)

        # 네트워크의 가중치 초기화
        initialize_weights(self.cls)

    # 순전파 함수 정의
    def forward(self, x):
        x2, x3, fea = self.model(x)  # ResNet 백본을 통해 특징 맵 추출

        if self.use_aux:
            seg_out = self.seg_head(x2, x3, fea)  # 세그멘테이션 출력 생성

        fea = self.pool(fea)  # 특징 맵 차원 축소

        # fea의 차원 변환 (벡터화)
        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)  ########################################실제 예측 하는곳 # 완전 연결층 통과하여 최종 출력 생성

        # 예측값을 딕셔너리 형태로 변환하여 반환
        pred_dict = {
            'loc_row': out[:, :self.dim1].view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
            'loc_col': out[:, self.dim1:self.dim1 + self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
            'exist_row': out[:, self.dim1 + self.dim2:self.dim1 + self.dim2 + self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row),
            'exist_col': out[:, -self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)
        }

        # 보조 네트워크 사용 시 세그멘테이션 결과 추가
        if self.use_aux:
            pred_dict['seg_out'] = seg_out

        return pred_dict  # 최종 결과 반환

    # Test-Time Augmentation(TTA) 적용한 순전파 함수
    def forward_tta(self, x):
        x2, x3, fea = self.model(x)  # ResNet 백본을 통해 특징 맵 추출

        pooled_fea = self.pool(fea)  # 특징 맵 축소
        n, c, h, w = pooled_fea.shape  # 차원 정보 저장

        # 여러 방향에서 특징 맵을 이동하여 TTA 적용
        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)

        left_pooled_fea[:, :, :, :w - 1] = pooled_fea[:, :, :, 1:]
        left_pooled_fea[:, :, :, -1] = pooled_fea.mean(-1)

        right_pooled_fea[:, :, :, 1:] = pooled_fea[:, :, :, :w - 1]
        right_pooled_fea[:, :, :, 0] = pooled_fea.mean(-1)

        up_pooled_fea[:, :, :h - 1, :] = pooled_fea[:, :, 1:, :]
        up_pooled_fea[:, :, -1, :] = pooled_fea.mean(-2)

        down_pooled_fea[:, :, 1:, :] = pooled_fea[:, :, :h - 1, :]
        down_pooled_fea[:, :, 0, :] = pooled_fea.mean(-2)

        # 모든 TTA 결과를 결합하여 특징 맵 생성
        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea, up_pooled_fea, down_pooled_fea], dim=0)
        fea = fea.view(-1, self.input_dim)  # 벡터 변환

        out = self.cls(fea)  # 최종 예측

        return {
            'loc_row': out[:, :self.dim1].view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
            'loc_col': out[:, self.dim1:self.dim1 + self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
            'exist_row': out[:, self.dim1 + self.dim2:self.dim1 + self.dim2 + self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row),
            'exist_col': out[:, -self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)
        }

# 모델 객체를 생성하는 함수 parsingNet(...)을 호출하여 모델을 생성한다.
#cfg(설정 파일)에서 필요한 설정값을 parsingNet의 매개변수로 전달한다.
#생성된 parsingNet 모델을 .cuda()를 사용하여 GPU에 올린 후 반환한다.
def get_model(cfg):
    return parsingNet(
        pretrained=True, 
        backbone=cfg.backbone, 
        num_grid_row=cfg.num_cell_row, 
        num_cls_row=cfg.num_row, 
        num_grid_col=cfg.num_cell_col, 
        num_cls_col=cfg.num_col, 
        num_lane_on_row=cfg.num_lanes, 
        num_lane_on_col=cfg.num_lanes, 
        use_aux=cfg.use_aux, 
        input_height=cfg.train_height, 
        input_width=cfg.train_width, 
        fc_norm=cfg.fc_norm
    ).cuda()  # 모델을 CUDA(GPU)로 로드
