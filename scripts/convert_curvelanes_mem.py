import os
import cv2
import tqdm
import json
import numpy as np
import json, argparse
import imagesize
import gc

def calc_k(line, height, width, angle=False):
    """
    Calculate the direction of lanes
    """
    line_x = line[::2]
    line_y = line[1::2]

    length = np.sqrt((line_x[0]-line_x[-1])**2 + (line_y[0]-line_y[-1])**2)
    if length < 90:
        return -10
    p = np.polyfit(line_x, line_y, deg=1)
    rad = np.arctan(p[0])

    if angle:
        return rad

    try:
        curve = np.polyfit(line_x[:2], line_y[:2], deg=1)
    except Exception:
        curve = np.polyfit(line_x[:3], line_y[:3], deg=1)

    try:
        curve1 = np.polyfit(line_y[:2], line_x[:2], deg=1)
    except Exception:
        curve1 = np.polyfit(line_y[:3], line_x[:3], deg=1)

    if rad < 0:
        y = np.poly1d(curve)(0)
        if y > height:
            result = np.poly1d(curve1)(height)
        else:
            result = -(height - y)
    else:
        y = np.poly1d(curve)(width)
        if y > height:
            result = np.poly1d(curve1)(height)
        else:
            result = width + (height - y)
    
    return result

def draw(im, line, idx, ratio_height=1, ratio_width=1, show=False):
    """
    Generate the segmentation label according to json annotation
    """
    line_x = np.array(line[::2]) * ratio_width
    line_y = np.array(line[1::2]) * ratio_height
    pt0 = (int(line_x[0]), int(line_y[0]))
    if show:
        cv2.putText(im, str(idx),
                    (int(line_x[len(line_x) // 2]), int(line_y[len(line_x) // 2]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 60

    for i in range(len(line_x) - 1):
        cv2.line(im, pt0,
                 (int(line_x[i + 1]), int(line_y[i + 1])),
                 (idx,), thickness=16)
        pt0 = (int(line_x[i + 1]), int(line_y[i + 1]))

def spline(arr, the_anno_row_anchor, ratio_height=1, ratio_width=1):
    arr = np.array(arr)
    arr[1::2] = arr[1::2] * ratio_height
    arr[::2]  = arr[::2]  * ratio_width
    curve = np.polyfit(arr[1::2], arr[::2], min(len(arr[::2]) - 1, 3))
    _min = arr[1::2].min()
    _max = arr[1::2].max()
    valid = ~((the_anno_row_anchor <= _max) & (the_anno_row_anchor >= _min))
    new_x = np.polyval(curve, the_anno_row_anchor)
    final_anno_list = np.concatenate([new_x.reshape(-1, 1), the_anno_row_anchor.reshape(-1, 1)], -1)
    final_anno_list[valid, 0] = -99999
    return final_anno_list

def generate_segmentation_and_train_list_streaming(
    dataset_root,
    label_dir='train',
    file_name='train_gt.txt',
    json_name='curvelanes_anno_cache.json',
    prefix_in_txt='train/'):
    """
    메모리를 최소화하기 위해 한 장씩 즉시 처리하고,
    JSON 결과도 한 번에 dict로 쌓지 않고 바로 파일에 streaming 방식으로 작성.
    """
    import math

    full_label_dir = os.path.join(dataset_root, label_dir)    # 예: .../train
    images_dir = os.path.join(full_label_dir, 'images')
    labels_dir = os.path.join(full_label_dir, 'labels')

    # 결과 세그멘트 이미지 저장 디렉토리
    seg_dir = os.path.join(full_label_dir, 'segs')
    if not os.path.exists(seg_dir):
        os.mkdir(seg_dir)

    train_gt_path = os.path.join(full_label_dir, file_name)
    train_gt_fp = open(train_gt_path, 'w')

    # JSON을 streaming으로 쓰기 위해 미리 파일을 열고 "{", "}"만 수동으로 관리
    json_out_path = os.path.join(full_label_dir, json_name)
    f_json = open(json_out_path, 'w')
    f_json.write('{')   # JSON 객체 시작
    is_first_json_item = True

    img_files = os.listdir(images_dir)

    for img_name in tqdm.tqdm(img_files):
        # 이미지와 라벨의 경로
        img_path_rel = os.path.join('images', img_name)  # ex) images/xxx.jpg
        img_path_abs = os.path.join(images_dir, img_name)

        label_json_name = img_name.replace('jpg', 'lines.json')
        label_json_path = os.path.join(labels_dir, label_json_name)

        # 라벨 json 불러오기
        with open(label_json_path, 'r') as f:
            lines_data = json.load(f)['Lines']

        # (x, y) 리스트로 정리
        temp_lines = []
        for line_points in lines_data:
            # y 내림차순 (기존 코드 동일)
            sorted_line = sorted(line_points, key=lambda x: -float(x['y']))
            temp_line = []
            for point in sorted_line:
                temp_line.append(float(point['x']))
                temp_line.append(float(point['y']))
            temp_lines.append(temp_line)

        # 이미지 크기
        width, height = imagesize.get(img_path_abs)

        # 각 라인 방향 계산
        ks = np.array([calc_k(line, height, width) for line in temp_lines], dtype=np.float32)
        ks_theta = np.array([calc_k(line, height, width, angle=True) for line in temp_lines], dtype=np.float32)

        # theta < 0, > 0 나누기
        k_neg = ks[ks_theta < 0].copy()
        k_neg_theta = ks_theta[ks_theta < 0].copy()
        k_pos = ks[ks_theta > 0].copy()
        k_pos_theta = ks_theta[ks_theta > 0].copy()

        # 너무 짧아서 -10인 것 제외
        k_neg = k_neg[k_neg_theta != -10]
        k_pos = k_pos[k_pos_theta != -10]

        # 정렬
        k_neg.sort()
        k_pos.sort()

        # 세그멘테이션 이미지
        label_img = np.zeros((height, width), dtype=np.uint8)

        bin_label = [0] * 10
        all_points = np.zeros((10, 125, 2), dtype=np.float32)
        the_anno_row_anchor = np.array(list(range(200, 1450, 10)))
        all_points[:, :, 1] = np.tile(the_anno_row_anchor, (10, 1))
        all_points[:, :, 0] = -99999

        # 리사이즈 비율
        rw = 2560 / width
        rh = 1440 / height

        # 왼쪽 방향 (k_neg)
        for idx in range(len(k_neg))[:5]:
            which_lane = np.where(ks == k_neg[idx])[0][0]
            draw(label_img, temp_lines[which_lane], 5 - idx)
            bin_label[4 - idx] = 1
            all_points[4 - idx] = spline(temp_lines[which_lane],
                                         the_anno_row_anchor,
                                         ratio_height=rh,
                                         ratio_width=rw)

        # 오른쪽 방향 (k_pos)
        for idx in range(len(k_pos))[:5]:
            which_lane = np.where(ks == k_pos[-(idx + 1)])[0][0]
            draw(label_img, temp_lines[which_lane], 6 + idx)
            bin_label[5 + idx] = 1
            all_points[5 + idx] = spline(temp_lines[which_lane],
                                         the_anno_row_anchor,
                                         ratio_height=rh,
                                         ratio_width=rw)

        # seg 이미지 저장
        seg_name = img_name[:-3] + 'png'  # xxx.png
        seg_path_rel = os.path.join('segs', seg_name)
        seg_path_abs = os.path.join(seg_dir, seg_name)
        cv2.imwrite(seg_path_abs, label_img)

        # train_gt.txt에 한 줄 기록
        train_gt_fp.write(prefix_in_txt + img_path_rel + ' ' +
                          prefix_in_txt + seg_path_rel + ' ' +
                          ' '.join(map(str, bin_label)) + '\n')

        # JSON에 바로 기록 (streaming)
        # key는 "train/images/xxx.jpg" 등
        json_key = prefix_in_txt + img_path_rel
        json_val = all_points.tolist()  # 10 x 125 x 2

        # 구분자 처리 (첫 아이템이 아니라면 콤마 추가)
        if not is_first_json_item:
            f_json.write(',')
        else:
            is_first_json_item = False

        # "key": [ ... ] 형태로 기록
        # key/val 모두 json.dumps 이용
        f_json.write(json.dumps(json_key))  # "train/images/xxx.jpg"
        f_json.write(':')
        f_json.write(json.dumps(json_val))  # [[ [x, y], [x, y], ... ], ...]

        # 메모리 절감을 위해 매 루프 종료 시 gc 처리
        del lines_data
        del temp_lines
        del label_img
        del ks
        del ks_theta
        del k_neg
        del k_neg_theta
        del k_pos
        del k_pos_theta
        gc.collect()

    # train_gt.txt, JSON 마무리
    train_gt_fp.close()
    f_json.write('}')
    f_json.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the CurveLanes dataset')
    return parser

if __name__ == "__main__":
    args = get_args().parse_args()

    # train 세트
    generate_segmentation_and_train_list_streaming(
        dataset_root=args.root,
        label_dir='train',
        file_name='train_gt.txt',
        json_name='curvelanes_anno_cache.json',
        prefix_in_txt='train/'
    )

    # valid 세트 (필요하다면 주석 해제)
    # generate_segmentation_and_train_list_streaming(
    #     dataset_root=args.root,
    #     label_dir='valid',
    #     file_name='valid_gt.txt',
    #     json_name='culane_anno_cache_val.json',
    #     prefix_in_txt='valid/'
    # )
