import os, argparse  # OS 및 명령줄 인자 파싱 라이브러리
from data.dali_data import TrainCollect  # 데이터 로드 관련 클래스
from utils.dist_utils import (get_rank, get_world_size, is_main_process, 
                              dist_print, DistSummaryWriter)  # 분산 처리 유틸리티 함수 및 로깅 클래스
from utils.config import Config  # 설정 파일을 로드하는 클래스
import torch  # PyTorch 라이브러리
import time  # 시간 측정 관련 라이브러리


# 문자열을 boolean 값으로 변환하는 함수
def str2bool(v):
    if isinstance(v, bool):  # 이미 bool 타입이면 그대로 반환
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):  # 'yes', 'true', 't' 등의 문자열을 True로 변환
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):  # 'no', 'false', 'f' 등의 문자열을 False로 변환
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')  # 변환할 수 없는 값이면 예외 발생


# 명령줄 인자를 처리하는 함수
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')  # 설정 파일 경로 필수 인자
    parser.add_argument('--local_rank', type=int, default=0)  # 분산 학습에서 로컬 rank 설정

    # 다양한 하이퍼파라미터 및 설정값을 명령줄 인자로 받을 수 있도록 설정
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--data_root', default=None, type=str)
    parser.add_argument('--epoch', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--optimizer', default=None, type=str)
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--momentum', default=None, type=float)
    parser.add_argument('--scheduler', default=None, type=str)
    parser.add_argument('--steps', default=None, type=int, nargs='+')  # 다중 step을 받을 수 있도록 설정
    parser.add_argument('--gamma', default=None, type=float)
    parser.add_argument('--warmup', default=None, type=str)
    parser.add_argument('--warmup_iters', default=None, type=int)
    parser.add_argument('--backbone', default=None, type=str)
    parser.add_argument('--griding_num', default=None, type=int)
    parser.add_argument('--use_aux', default=None, type=str2bool)  # boolean 변환
    parser.add_argument('--sim_loss_w', default=None, type=float)
    parser.add_argument('--shp_loss_w', default=None, type=float)
    parser.add_argument('--note', default=None, type=str)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--finetune', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--test_model', default=None, type=str)
    parser.add_argument('--test_work_dir', default=None, type=str)
    parser.add_argument('--num_lanes', default=None, type=int)
    parser.add_argument('--auto_backup', action='store_false', help='자동 백업 비활성화')  # 기본적으로 활성화됨
    parser.add_argument('--var_loss_power', default=None, type=float)
    parser.add_argument('--num_row', default=None, type=int)
    parser.add_argument('--num_col', default=None, type=int)
    parser.add_argument('--train_width', default=None, type=int)
    parser.add_argument('--train_height', default=None, type=int)
    parser.add_argument('--num_cell_row', default=None, type=int)
    parser.add_argument('--num_cell_col', default=None, type=int)
    parser.add_argument('--mean_loss_w', default=None, type=float)
    parser.add_argument('--fc_norm', default=None, type=str2bool)
    parser.add_argument('--soft_loss', default=None, type=str2bool)
    parser.add_argument('--cls_loss_col_w', default=None, type=float)
    parser.add_argument('--cls_ext_col_w', default=None, type=float)
    parser.add_argument('--mean_loss_col_w', default=None, type=float)
    parser.add_argument('--eval_mode', default=None, type=str)
    parser.add_argument('--eval_during_training', default=None, type=str2bool)
    parser.add_argument('--split_channel', default=None, type=str2bool)
    parser.add_argument('--match_method', default=None, type=str, choices=['fixed', 'hungarian'])  # 매칭 방법 선택
    parser.add_argument('--selected_lane', default=None, type=int, nargs='+')  # 여러 차선 선택 가능
    parser.add_argument('--cumsum', default=None, type=str2bool)
    parser.add_argument('--masked', default=None, type=str2bool)
    
    return parser  # ArgumentParser 객체 반환


import numpy as np  # 수치 계산 라이브러리

# 명령줄 인자를 설정 파일과 병합하는 함수
def merge_config():
    args = get_args().parse_args()  # 명령줄 인자 파싱
    cfg = Config.fromfile(args.config)  # 설정 파일 불러오기

    # 명령줄 인자로 설정할 수 있는 항목 리스트
    items = ['dataset', 'data_root', 'epoch', 'batch_size', 'optimizer', 'learning_rate',
             'weight_decay', 'momentum', 'scheduler', 'steps', 'gamma', 'warmup', 'warmup_iters',
             'use_aux', 'griding_num', 'backbone', 'sim_loss_w', 'shp_loss_w', 'note', 'log_path',
             'finetune', 'resume', 'test_model', 'test_work_dir', 'num_lanes', 'var_loss_power',
             'num_row', 'num_col', 'train_width', 'train_height', 'num_cell_row', 'num_cell_col',
             'mean_loss_w', 'fc_norm', 'soft_loss', 'cls_loss_col_w', 'cls_ext_col_w', 'mean_loss_col_w',
             'eval_mode', 'eval_during_training', 'split_channel', 'match_method', 'selected_lane',
             'cumsum', 'masked']

    # 명령줄에서 제공된 값이 있으면 설정값을 덮어쓰기
    for item in items:
        if getattr(args, item) is not None:
            dist_print('merge ', item, ' config')  # 병합된 설정 로그 출력
            setattr(cfg, item, getattr(args, item))  # 설정값 업데이트

    # 데이터셋에 따라 row_anchor, col_anchor 값 설정########################################################
    if cfg.dataset == 'CULane':
        cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)  # 차선 위치에 대한 정규화 값
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == 'Tusimple':
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720  # Tusimple 데이터셋 기준 정규화
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == 'CurveLanes':
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)  # CurveLanes용 차선 정규화
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    
    return args, cfg  # 업데이트된 설정값 반환


# 모델을 저장하는 함수
def save_model(net, optimizer, epoch, save_path, distributed):
    if is_main_process():  # 메인 프로세스에서만 실행
        model_state_dict = net.state_dict()  # 모델의 상태 저장
        state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}  # 모델과 옵티마이저 상태 저장
        assert os.path.exists(save_path)  # 저장 경로가 존재하는지 확인
        model_path = os.path.join(save_path, 'model_best.pth')  # 모델 저장 경로 설정
        torch.save(state, model_path)  # 모델 저장


import pathspec  # 파일 및 디렉토리 필터링을 위한 라이브러리

# 프로젝트 디렉토리를 백업하는 함수
def cp_projects(auto_backup, to_path):
    if is_main_process() and auto_backup:  # 메인 프로세스이고 자동 백업이 활성화된 경우 실행
        with open('./.gitignore', 'r') as fp:  # .gitignore 파일 읽기
            ign = fp.read()
        ign += '\n.git'  # Git 디렉토리 제외

        # .gitignore 패턴을 사용하여 제외할 파일 리스트 생성
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
        all_files = {os.path.join(root, name) for root, dirs, files in os.walk('./') for name in files}  # 모든 파일 수집
        matches = spec.match_files(all_files)  # 제외할 파일 필터링
        matches = set(matches)
        to_cp_files = all_files - matches  # 복사할 파일 리스트

        dist_print('Copying projects to ' + to_path + ' for backup')  # 복사 시작 로그 출력
        t0 = time.time()  # 시작 시간 기록
        warning_flag = True

        for f in to_cp_files:  # 백업할 파일 순회
            dirs = os.path.join(to_path, 'code', os.path.split(f[2:])[0])  # 대상 경로 생성
            if not os.path.exists(dirs):
                os.makedirs(dirs)  # 필요한 디렉토리 생성
            os.system('cp %s %s' % (f, os.path.join(to_path, 'code', f[2:])))  # 파일 복사

            elapsed_time = time.time() - t0  # 경과 시간 계산
            if elapsed_time > 5 and warning_flag:  # 복사 시간이 길어질 경우 경고 출력
                dist_print('If the program is stuck, it might be copying large files in this directory. '
                           'please don\'t set --auto_backup. Or please make your working directory clean, '
                           'i.e., don\'t place large files like dataset, log results under this directory.')
                warning_flag = False  # 경고 메시지는 한 번만 출력


import datetime, os  # 날짜 및 OS 관련 라이브러리

# 작업 디렉토리 생성 함수
def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 현재 날짜 및 시간 포맷팅
    hyper_param_str = '_lr_%1.0e_b_%d' % (cfg.learning_rate, cfg.batch_size)  # 학습률 및 배치 크기 포함
    work_dir = os.path.join(cfg.log_path, now + hyper_param_str + cfg.note)  # 로그 경로 생성
    return work_dir


# 로깅을 위한 SummaryWriter 초기화 함수
def get_logger(work_dir, cfg):
    logger = DistSummaryWriter(work_dir)  # 메인 프로세스에서 TensorBoard 로깅
    config_txt = os.path.join(work_dir, 'cfg.txt')

    if is_main_process():  # 메인 프로세스에서만 설정 파일 저장
        with open(config_txt, 'w') as fp:
            fp.write(str(cfg))

    return logger  # 로거 반환


# 모델 가중치 초기화 함수
def initialize_weights(*models):
    for model in models:
        real_init_weights(model)  # 모델의 가중치 초기화


# 실제 가중치 초기화 수행 함수
def real_init_weights(m):
    if isinstance(m, list):  # 모델이 리스트인 경우 각 모델에 대해 초기화 수행
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):  # 합성곱 레이어 초기화
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Kaiming 초기화 적용
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)  # bias 초기화
        elif isinstance(m, torch.nn.Linear):  # 완전연결 레이어 초기화
            m.weight.data.normal_(0.0, std=0.01)  # 정규 분포 초기화
        elif isinstance(m, torch.nn.BatchNorm2d):  # 배치 정규화 레이어 초기화
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Module):  # 재귀적으로 하위 모듈 초기화
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unknown module', m)  # 알 수 없는 모듈 경고 출력


import importlib  # 동적 모듈 로딩을 위한 라이브러리

# 데이터셋에 맞는 모델 불러오는 함수  ✅ cfg.dataset.lower() 값을 이용하여 해당 데이터셋에 맞는 model_~~~.py 파일을 임포트하고, 그 안에 있는 get_model(cfg) 함수를 호출한다.
def get_model(cfg):
    return importlib.import_module('model.model_' + cfg.dataset.lower()).get_model(cfg)


# 학습 데이터 로더 생성 함수
def get_train_loader(cfg):
    if cfg.dataset == 'CULane':  # CULane 데이터셋 학습 로더
        train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'list/train_gt.txt'), get_rank(), get_world_size(),
                                    cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, cfg.dataset, cfg.crop_ratio)
    elif cfg.dataset == 'Tusimple':  # Tusimple 데이터셋 학습 로더
        train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'train_gt.txt'), get_rank(), get_world_size(),
                                    cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, cfg.dataset, cfg.crop_ratio)
    elif cfg.dataset == 'CurveLanes':  # CurveLanes 데이터셋 학습 로더
        train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'train', 'train_gt.txt'), get_rank(), get_world_size(),
                                    cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, cfg.dataset, cfg.crop_ratio)
    else:
        raise NotImplementedError  # 지원되지 않는 데이터셋 예외 발생
    return train_loader  # 학습 데이터 로더 반환


# 데이터셋에 따른 추론 함수
def inference(net, data_label, dataset):
    if dataset == 'CurveLanes':
        return inference_curvelanes(net, data_label)
    elif dataset in ['Tusimple', 'CULane']:
        return inference_culane_tusimple(net, data_label)
    else:
        raise NotImplementedError  # 지원되지 않는 데이터셋 예외 발생


# Tusimple 및 CULane 데이터셋에서 추론 수행 함수
def inference_culane_tusimple(net, data_label):
    pred = net(data_label['images'])  # 모델 예측 수행 , train이미지인데 data_label['images']는 원본 이미지가 아니라, 이미 decode + normalize + resize 완료된 **전처리된 GPU 텐서(batch)**입니다.
    cls_out_ext_label = (data_label['labels_row'] != -1).long()  # 차선 존재 여부를 나타내는 바이너리 행렬,차선의 위치를 의미하는 GT(정답) 데이터, 유효한 행(label) 필터링
    cls_out_col_ext_label = (data_label['labels_col'] != -1).long()  # 차선 존재 여부를 나타내는 바이너리 행렬 ,차선의 위치를 의미하는 GT(정답) 데이터,유효한 열(label) 필터링

    res_dict = {'cls_out': pred['loc_row'], 'cls_label': data_label['labels_row'], 
                'cls_out_col': pred['loc_col'], 'cls_label_col': data_label['labels_col'],
                'cls_out_ext': pred['exist_row'], 'cls_out_ext_label': cls_out_ext_label, 
                'cls_out_col_ext': pred['exist_col'], 'cls_out_col_ext_label': cls_out_col_ext_label, 
                'labels_row_float': data_label['labels_row_float'], 'labels_col_float': data_label['labels_col_float']}
    
    if 'seg_out' in pred.keys():  # 분할(segmentation) 결과가 있을 경우 추가
        res_dict['seg_out'] = pred['seg_out']
        res_dict['seg_label'] = data_label['seg_images']

    return res_dict


# CurveLanes 데이터셋에서 추론 수행 함수
def inference_curvelanes(net, data_label):
    pred = net(data_label['images']) #model_curvelanes.py의  forward() 호출 
    cls_out_ext_label = (data_label['labels_row'] != -1).long().cuda()
    cls_out_col_ext_label = (data_label['labels_col'] != -1).long().cuda()

    res_dict = {'cls_out': pred['loc_row'], 'cls_label': data_label['labels_row'].cuda(), 
                'cls_out_col': pred['loc_col'], 'cls_label_col': data_label['labels_col'].cuda(),
                'cls_out_ext': pred['exist_row'], 'cls_out_ext_label': cls_out_ext_label, 
                'cls_out_col_ext': pred['exist_col'], 'cls_out_col_ext_label': cls_out_col_ext_label, 
                'seg_label': data_label['seg_images'].cuda(), 'seg_out_row': pred['lane_token_row'], 
                'seg_out_col': pred['lane_token_col']}
    
    if 'seg_out' in pred.keys():
        res_dict['seg_out'] = pred['seg_out']
        res_dict['seg_label'] = data_label['segs']
        
    # # ✅ 텐서 저장 함수 정의
    # def save_tensor(tensor, filename):
    #     tensor_cpu = tensor.detach().cpu()
    #     with open(filename, "w") as f:
    #         f.write(f"# shape: {list(tensor_cpu.shape)}\n")
    #         f.write(str(tensor_cpu))

    # # ✅ 텐서들 저장
    # for name, tensor in res_dict.items():
    #     save_tensor(tensor, f"{name}.txt")

    # # ✅ 대기
    # time.sleep(10)
    # print("end")
    
    return res_dict #train.py results로 들어감

# 손실(loss) 계산 함수
def calc_loss(loss_dict, results, logger, global_step, epoch):
    loss = 0  # 총 손실값 초기화

    # 손실 항목 개수만큼 반복
    for i in range(len(loss_dict['name'])):

        if loss_dict['weight'][i] == 0:  # 가중치가 0이면 해당 손실 무시
            continue
            
        data_src = loss_dict['data_src'][i]  # 현재 손실 계산에 필요한 데이터 소스 추출

        datas = [results[src] for src in data_src]  # results에서 해당 데이터 가져오기

        loss_cur = loss_dict['op'][i](*datas)  # 손실 함수 적용하여 현재 손실값 계산

        if global_step % 20 == 0:  # 20 스텝마다 로그 기록
            logger.add_scalar('loss/' + loss_dict['name'][i], loss_cur, global_step)  # TensorBoard에 손실 기록

        loss += loss_cur * loss_dict['weight'][i]  # 손실값을 가중치와 함께 총 손실값에 더하기

    return loss  # 최종 손실값 반환
