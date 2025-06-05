import torch
import torch.distributed as dist  # PyTorch 분산 처리(distributed training) 관련 모듈
import pickle  # 객체 직렬화 및 역직렬화를 위한 라이브러리


# 현재 실행 중인 프로세스가 속한 분산 학습 환경의 전체 프로세스 수(월드 사이즈)를 반환
def get_world_size():
    if not dist.is_available():  # 분산 처리가 사용 가능한지 확인
        return 1
    if not dist.is_initialized():  # 분산 처리가 초기화되었는지 확인
        return 1
    return dist.get_world_size()  # 전체 프로세스 개수 반환


# PyTorch 텐서를 float 값으로 변환하는 함수
def to_python_float(t):
    if hasattr(t, 'item'):  # 텐서에 item() 메서드가 존재하면 해당 값 반환
        return t.item()
    else:
        return t[0]  # 리스트 형태라면 첫 번째 원소 반환


# 현재 실행 중인 프로세스의 rank(고유 ID) 반환
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()  # 현재 rank 반환


# 현재 프로세스가 메인 프로세스(랭크 0)인지 확인하는 함수
def is_main_process():
    return get_rank() == 0


# 현재 프로세스에서 로그를 출력할 수 있는지 확인하는 함수
def can_log():
    return is_main_process()


# 메인 프로세스에서만 로그를 출력하는 함수
def dist_print(*args, **kwargs):
    if can_log():
        print(*args, **kwargs)


# 모든 분산 프로세스를 동기화하는 함수 (barrier 역할 수행)
def synchronize():
    """
    분산 학습 시 모든 프로세스를 동기화하는 함수
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:  # 단일 프로세스라면 동기화할 필요 없음
        return
    dist.barrier()  # 모든 프로세스가 여기서 멈추고 동기화될 때까지 기다림


# 모든 프로세스의 텐서를 모아서 연결하는 함수
def dist_cat_reduce_tensor(tensor):
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()  # 입력 텐서 복사
    all_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]  # 프로세스 수만큼 텐서 리스트 생성
    dist.all_gather(all_list, rt)  # 모든 프로세스에서 텐서 수집
    return torch.cat(all_list, dim=0)  # 모든 텐서를 연결하여 반환


# 모든 프로세스에서 텐서를 합산하는 함수
def dist_sum_reduce_tensor(tensor):
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor
    if not isinstance(tensor, torch.Tensor):  # 입력이 텐서인지 확인
        return tensor
    rt = tensor.clone()  # 입력 텐서 복사
    dist.all_reduce(rt, op=dist.reduce_op.SUM)  # 모든 프로세스에서 값을 합산
    return rt


# 모든 프로세스에서 평균 값을 구하는 함수
def dist_mean_reduce_tensor(tensor):
    rt = dist_sum_reduce_tensor(tensor)  # 모든 프로세스의 값을 합산
    rt /= get_world_size()  # 프로세스 수로 나누어 평균값 계산
    return rt


# 모든 프로세스에서 데이터(객체)를 수집하는 함수
def all_gather(data):
    """
    모든 프로세스에서 데이터를 수집하는 함수 (객체도 가능)
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]  # 단일 프로세스라면 그냥 리스트로 반환

    buffer = pickle.dumps(data)  # 데이터를 직렬화
    storage = torch.ByteStorage.from_buffer(buffer)  # 바이트 스토리지 생성
    tensor = torch.ByteTensor(storage).to("cuda")  # CUDA 텐서로 변환

    # 각 프로세스에서 데이터 크기 구하기
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)  # 모든 프로세스의 데이터 크기 수집
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)  # 최대 크기 결정

    # 모든 프로세스의 데이터를 동일한 크기로 맞춰서 수집
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:  # 데이터 크기가 다르면 패딩 추가
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)  # 모든 프로세스의 데이터를 수집

    data_list = []
    for size, tensor in zip(size_list, tensor_list):  # 수집된 데이터 복원
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list  # 모든 프로세스의 데이터를 리스트 형태로 반환


from torch.utils.tensorboard import SummaryWriter  # TensorBoard 로깅을 위한 라이브러리


# 분산 환경에서도 메인 프로세스에서만 TensorBoard 기록을 수행하는 클래스
class DistSummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        if can_log():  # 메인 프로세스일 경우에만 TensorBoard 초기화
            super(DistSummaryWriter, self).__init__(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        if can_log():  # 메인 프로세스에서만 스칼라 값 기록
            super(DistSummaryWriter, self).add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if can_log():  # 메인 프로세스에서만 이미지 기록
            super(DistSummaryWriter, self).add_figure(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        if can_log():  # 메인 프로세스에서만 그래프 기록
            super(DistSummaryWriter, self).add_graph(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if can_log():  # 메인 프로세스에서만 히스토그램 기록
            super(DistSummaryWriter, self).add_histogram(*args, **kwargs)
    
    def add_image(self, *args, **kwargs):
        if can_log():  # 메인 프로세스에서만 이미지 기록
            super(DistSummaryWriter, self).add_image(*args, **kwargs)

    def close(self):
        if can_log():  # 메인 프로세스에서만 로그 파일 닫기
            super(DistSummaryWriter, self).close()


import tqdm  # 진행률 표시 라이브러리


# 분산 환경에서도 메인 프로세스에서만 tqdm(진행 상태바) 사용
def dist_tqdm(obj, *args, **kwargs):
    if can_log():
        return tqdm.tqdm(obj, *args, **kwargs)  # 메인 프로세스에서 tqdm 사용
    else:
        return obj  # 서브 프로세스에서는 tqdm을 사용하지 않고 원본 객체 반환
