import torch, os, datetime  # PyTorch, OS 및 날짜 관련 라이브러리

# 분산 학습 및 로깅 관련 유틸리티 함수들 불러오기
from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics
from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger
import time  # 시간 측정 라이브러리
from evaluation.eval_wrapper import eval_lane  # 차선 평가 함수 불러오기

def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, dataset):
    net.train()  # 모델을 학습 모드로 전환
    progress_bar = dist_tqdm(train_loader)  # tqdm을 이용한 진행 상태 출력
    for b_idx, data_label in enumerate(progress_bar):  # 배치 단위로 데이터 로드
        global_step = epoch * len(data_loader) + b_idx  # 현재 global step 계산

        results = inference(net, data_label, dataset)  # 모델 예측 수행 , model_curvelanes.py의  forward() 호출 

        loss = calc_loss(loss_dict, results, logger, global_step, epoch)  # 손실 계산
        optimizer.zero_grad()  # 이전 그래디언트 초기화
        loss.backward()  # 역전파 수행
        optimizer.step()  # 가중치 업데이트
        scheduler.step(global_step)  # 학습률 조정

        if global_step % 20 == 0:  # 20 스텝마다 로깅 및 메트릭 업데이트
            reset_metrics(metric_dict)  # 메트릭 초기화
            update_metrics(metric_dict, results)  # 메트릭 업데이트
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)  # 메트릭 로깅
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)  # 학습률 로깅

            if hasattr(progress_bar, 'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
                new_kwargs = {}
                for k, v in kwargs.items():
                    if 'lane' in k:
                        continue
                    new_kwargs[k] = v
                progress_bar.set_postfix(loss='%.3f' % float(loss), **new_kwargs)  # 진행 상태 업데이트

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # cuDNN 최적화 활성화
    args, cfg = merge_config()  # 설정 파일 불러오기

    if args.local_rank == 0:
        work_dir = get_work_dir(cfg)  # 작업 디렉토리 생성

    distributed = False  # 분산 학습 여부 확인
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)  # GPU 설정
        torch.distributed.init_process_group(backend='nccl', init_method='env://')  # 분산 학습 초기화

        if args.local_rank == 0:
            with open('.work_dir_tmp_file.txt', 'w') as f:
                f.write(work_dir)  # 작업 디렉토리 경로 저장
        else:
            while not os.path.exists('.work_dir_tmp_file.txt'):
                time.sleep(0.1)
            with open('.work_dir_tmp_file.txt', 'r') as f:
                work_dir = f.read().strip()

    synchronize()  # 모든 프로세스 동기화
    cfg.test_work_dir = work_dir
    cfg.distributed = distributed

    if args.local_rank == 0:
        os.system('rm .work_dir_tmp_file.txt')  # 임시 파일 삭제
    
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide', '34fca']  # 유효한 백본 확인

    train_loader = get_train_loader(cfg)  # 학습 데이터 로더 생성
    net = get_model(cfg)  # 모델 생성

    if distributed:
        # 모델을 분산 학습 모드로 설정 (멀티 GPU 사용 시 필요)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    # 옵티마이저(최적화 함수) 설정 (SGD, Adam 등)
    optimizer = get_optimizer(net, cfg)

    # 사전 학습된 모델을 불러와서 미세 조정(finetuning)할 경우
    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)  # finetune 모델을 불러온다고 출력
        state_all = torch.load(cfg.finetune)['model']  # 사전 학습된 모델의 가중치를 로드
        state_clip = {}  # 백본(backbone) 부분만 로드할 딕셔너리 생성
        
        for k, v in state_all.items():  # 가중치 딕셔너리를 순회
            if 'model' in k:  # 'model' 키를 포함하는 항목만 선택
                state_clip[k] = v  # 선택한 가중치를 새로운 딕셔너리에 저장
        
        net.load_state_dict(state_clip, strict=False)  # 모델에 가중치 적용 (일부만 적용 가능하도록 strict=False 설정)

    # 학습을 이어서 진행(resume)하는 경우
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)  # 재개할 모델 경로 출력
        resume_dict = torch.load(cfg.resume, map_location='cpu')  # 체크포인트 파일 로드 (CPU에 로드)
        net.load_state_dict(resume_dict['model'])  # 모델 가중치 적용
        
        # 옵티마이저 상태도 저장되어 있다면 불러오기
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])  # 옵티마이저 상태 복원
        
        # 체크포인트 파일명에서 epoch 숫자를 추출하여 다음 epoch 설정
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1  
    else:
        resume_epoch = 0  # 처음부터 학습을 시작


    scheduler = get_scheduler(optimizer, cfg, len(train_loader))  # 스케줄러 설정
    dist_print(len(train_loader))  # 데이터 로더 길이 출력
    metric_dict = get_metric_dict(cfg)  # 메트릭 초기화
    loss_dict = get_loss_dict(cfg)  # 손실 함수 설정
    logger = get_logger(work_dir, cfg)  # 로깅 설정
    max_res = 0  # 최고 결과 저장 변수
    res = None  # 평가 결과 변수

    for epoch in range(resume_epoch, cfg.epoch):
        dist_print(f"================ Epoch {epoch} 시작 ================")

        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.dataset)  # 학습 실행
        train_loader.reset()  # 데이터 로더 초기화

        res = eval_lane(net, cfg, ep=epoch, logger=logger)  # 모델 평가

        if res is not None and res > max_res:
            max_res = res
            save_model(net, optimizer, epoch, work_dir, distributed)  # 최고 성능 모델 저장
        logger.add_scalar('CuEval/X', max_res, global_step=epoch)  # 평가 결과 로깅

    logger.close()  # 로깅 종료
