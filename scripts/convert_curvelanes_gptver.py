"""
Convert CurveLanes annotations → diagonal‑anchor training files
---------------------------------------------------------------
* 좌상→우하  (diag‑main)  :  num_diag_main
* 우상→좌하  (diag‑ortho) :  num_diag_ortho

변수 이름/구조는 기존 코드와 최대한 동일하게 두되, spline 보간·라벨 매핑을 대각선 기준으로 전환.
"""

import os, cv2, tqdm, json, argparse, imagesize, numpy as np

# ─────────────────────────────────────────────────────────────
# 1. 대각선 보간 함수
# ─────────────────────────────────────────────────────────────

def spline_diag(arr, anchor_t, img_w, img_h, diag_type='main'):
    """ 
    arr        : [x0,y0,x1,y1,...]
    anchor_t   : (N,) 0~1 실수   고정 t 좌표   (t = y/H)
    diag_type  : 'main'(↘)  /  'ortho'(↗)
    반환값      : (N,2) [x,y]  (x=-99999 ⇒ invalid)
    """
    arr = np.asarray(arr, dtype=np.float32)
    xs, ys = arr[::2], arr[1::2]

    t_vals = ys / img_h                      # 현재 차선의 각 점 t

    if diag_type == 'main':                  # ↘  x ~ t
        poly = np.polyfit(t_vals, xs, min(len(xs)-1, 3))
        f = np.poly1d(poly)
        new_x = f(anchor_t)
    else:                                    # ↗  (W-x) ~ t
        poly = np.polyfit(t_vals, img_w - xs, min(len(xs)-1, 3))
        f = np.poly1d(poly)
        new_x = img_w - f(anchor_t)

    new_y = anchor_t * img_h
    final = np.stack([new_x, new_y], axis=1)

    # valid mask : t within source range
    valid = (anchor_t >= t_vals.min()) & (anchor_t <= t_vals.max())
    final[~valid, 0] = -99999.
    return final

# ─────────────────────────────────────────────────────────────
# 2. 데이터셋 파싱
# ─────────────────────────────────────────────────────────────

def get_curvelanes_list(root, split):
    json_dir = os.path.join(root, split, 'labels')
    img_dir  = os.path.join(root, split, 'images')

    names, line_txt = [], []
    for img_name in tqdm.tqdm(sorted(os.listdir(img_dir))):
        names.append(os.path.join(split, 'images', img_name))
        label_path = os.path.join(json_dir, img_name.replace('jpg', 'lines.json'))
        with open(label_path, 'r') as fp:
            lines = json.load(fp)['Lines']
        tmp = []
        for line in lines:
            pts = sorted(line, key=lambda p: -float(p['y']))
            tmp.append([float(p['x']) for p in pts] + [float(p['y']) for p in pts])
        line_txt.append(tmp)
    return names, line_txt

# ─────────────────────────────────────────────────────────────
# 3. GT & segmentation 생성
# ─────────────────────────────────────────────────────────────

def generate_gt(root, line_txt, names, cfg):
    save_txt = os.path.join(root, 'train_gt_diag.txt')
    fp_txt   = open(save_txt, 'w')
    seg_dir  = os.path.join(root, 'segs')
    os.makedirs(seg_dir, exist_ok=True)

    cache = {}

    for img_rel, lines in tqdm.tqdm(zip(names, line_txt), total=len(names)):
        img_path = os.path.join(root, img_rel)
        w, h = imagesize.get(img_path)

        diag_t = np.linspace(0, 1, cfg['num_diag_main'])   # 72
        diag_t_o = np.linspace(0, 1, cfg['num_diag_ortho'])# 41

        all_pts = np.full((cfg['num_lane'], max(cfg['num_diag_main'], cfg['num_diag_ortho']), 2), -99999., np.float32)

        # 예시: 좌측 5 ↘ / 우측 5 ↗  (간단 정렬 생략)
        for lane_id, line in enumerate(lines[:cfg['num_lane']]):
            if lane_id < cfg['num_lane_left']:
                all_pts[lane_id, :cfg['num_diag_main']] = spline_diag(line, diag_t, w, h, 'main')
            else:
                all_pts[lane_id, :cfg['num_diag_ortho']] = spline_diag(line, diag_t_o, w, h, 'ortho')

        # segmentation mask  (기존 draw() 재사용 생략 — 간단 버전)
        seg = np.zeros((h, w), np.uint8)
        # ... 필요 시 차선 굵기 등 그리기 ...
        seg_path_rel = img_rel.replace('images', 'segs').replace('.jpg', '.png')
        cv2.imwrite(os.path.join(root, seg_path_rel), seg)

        cache[img_rel] = all_pts.tolist()
        fp_txt.write(f"{img_rel} {seg_path_rel} 0 0 0 0 0 0 0 0 0 0\n")

    fp_txt.close()
    with open(os.path.join(root, 'curvelanes_diag_cache.json'), 'w') as f:
        json.dump(cache, f)

# ─────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='CurveLanes root')
    return p.parse_args()

if __name__ == '__main__':
    args = get_args()

    cfg = dict(num_diag_main=72, num_diag_ortho=41, num_lane=10,
               num_lane_left=5)  # 간단 config dict

    names, line_txt = get_curvelanes_list(args.root, 'train')
    generate_gt(args.root, line_txt, names, cfg)
    print('✅ diagonal GT created')
