"""
Fancam Studio v2 — core engine
兩段式 offline pipeline：
  Pass 1  BoT-SORT(+ReID, +GMC) 逐格追蹤 → 儲存所有 tracklets + 外觀特徵
  Stitch  離線用外觀相似度 + 時空 gating 縫合斷開嘅 tracklets
  Target  飯拍模式=面容識別 / 運動模式=外觀+手動 keyframe
  Report  逐格信心評分 → 準備度% + 標出要人手覆核嘅時段
  Render  任意比例、任意解像度、平滑運鏡、原聲
"""
import os
import json
import subprocess
import numpy as np
import cv2
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER_CFG = os.path.join(HERE, "botsort_reid.yaml")
TRACKER_CFG_FAST = os.path.join(HERE, "botsort_fast.yaml")


def make_tracking_proxy(video_path, height=720):
    """出一條細proxy專門做追蹤（原片留返 render 用）。4K 片 decode 本身就係大瓶頸。
    回傳 (proxy路徑, 座標放大倍數)"""
    import tempfile
    cap = cv2.VideoCapture(video_path)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if H <= height:
        return video_path, 1.0
    proxy = os.path.join(tempfile.gettempdir(),
                         os.path.splitext(os.path.basename(video_path))[0] + f"_track{height}.mp4")
    if not os.path.exists(proxy):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", video_path,
             "-vf", f"scale=-2:{height}", "-c:v", "libx264",
             "-preset", "ultrafast", "-crf", "20", "-an", proxy],
            check=True)
    return proxy, H / height


def scale_tracks(tracks, factor):
    if factor == 1.0:
        return tracks
    return {tid: {fi: tuple(v * factor for v in box) for fi, box in fr.items()}
            for tid, fr in tracks.items()}


def best_device():
    """自動揀最快裝置：Apple MPS > NVIDIA CUDA > CPU"""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# ---------------- models ----------------
_YOLO = None
_FACE_APP = None

def get_yolo(name="yolov8m.pt"):
    global _YOLO
    if _YOLO is None:
        _YOLO = YOLO(name)
    return _YOLO

def get_face_app():
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        _FACE_APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _FACE_APP.prepare(ctx_id=0, det_size=(640, 640))
    return _FACE_APP


def build_reference_embedding(image_paths):
    from ingest import load_image
    app = get_face_app()
    embs = []
    for p in image_paths:
        img = load_image(p)
        if img is None:
            continue
        faces = app.get(img)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            embs.append(f.normed_embedding)
    if not embs:
        return None
    ref = np.mean(embs, axis=0)
    return ref / np.linalg.norm(ref)


# ---------------- appearance embedding (full body, for sports) ----------------
def _body_embedding(frame, box):
    """全身外觀特徵：HSV 直方圖（上/下半身分開）— 唔使額外模型，CPU 都快"""
    x1, y1, x2, y2 = map(int, box)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (64, 128))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    parts = []
    for seg in (hsv[:64], hsv[64:]):  # 上身 / 下身
        h = cv2.calcHist([seg], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        h = h / (h.sum() + 1e-6)
        parts.append(h)
    v = np.concatenate(parts)
    return v / (np.linalg.norm(v) + 1e-6)


# ================= PASS 1: full-video tracking =================
def track_video(video_path, model_name="yolov8m.pt", conf=0.3, imgsz=960,
                progress_cb=None, sample_appearance_every=5,
                stride=2, device=None, tracker_cfg=None):
    """stride=N 即係每 N 格先偵測一次，中間插值（快 N 倍，人物移動連續所以夠準）"""
    if device is None:
        device = best_device()
    if tracker_cfg is None:
        tracker_cfg = TRACKER_CFG
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = get_yolo(model_name)
    model.predictor = None
    tracks = {}
    app_sums, app_cnts = {}, {}

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % stride != 0:
            fi += 1
            continue
        r = model.track(frame, classes=[0], persist=True, verbose=False,
                        conf=conf, imgsz=imgsz, tracker=tracker_cfg, device=device)[0]
        if r.boxes is not None and r.boxes.id is not None:
            for b in r.boxes:
                tid = int(b.id.item())
                box = tuple(b.xyxy[0].tolist())
                tracks.setdefault(tid, {})[fi] = box
                if fi % sample_appearance_every == 0:
                    emb = _body_embedding(frame, box)
                    if emb is not None:
                        app_sums[tid] = app_sums.get(tid, 0) + emb
                        app_cnts[tid] = app_cnts.get(tid, 0) + 1
        fi += 1
        if progress_cb and fi % 30 == 0:
            progress_cb(fi / max(total, 1), f"Pass 1 追蹤 {fi}/{total}")
    cap.release()

    # stride 插值：同一 tid 相鄰兩次偵測之間嘅格，線性補返（gap 細所以準）
    if stride > 1:
        for tid, fr in tracks.items():
            fs = sorted(fr)
            for a, b in zip(fs, fs[1:]):
                if 1 < b - a <= stride * 2:
                    ba, bb = np.array(fr[a]), np.array(fr[b])
                    for k in range(a + 1, b):
                        t = (k - a) / (b - a)
                        fr[k] = tuple(ba * (1 - t) + bb * t)

    apps = {}
    for tid, s in app_sums.items():
        v = s / app_cnts[tid]
        apps[tid] = v / (np.linalg.norm(v) + 1e-6)
    meta = {"W": W, "H": H, "fps": fps, "total": fi}
    return tracks, apps, meta


# ================= STITCH: offline tracklet merging =================
def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def stitch_tracklets(tracks, apps, meta, app_th=0.80, max_gap_sec=3.0, max_dist_frac=0.35):
    """縫合斷開嘅 tracklets：時間唔重疊 + gap 唔太長 + 位置接近 + 外觀相似"""
    fps = meta["fps"]
    diag = np.hypot(meta["W"], meta["H"])
    info = {}
    for tid, fr in tracks.items():
        fs = sorted(fr)
        info[tid] = {"start": fs[0], "end": fs[-1],
                     "start_c": _center(fr[fs[0]]), "end_c": _center(fr[fs[-1]])}
    tids = sorted(info, key=lambda t: info[t]["start"])
    parent = {t: t for t in tids}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for i, a in enumerate(tids):
        for b in tids[i+1:]:
            ia, ib = info[a], info[b]
            gap = ib["start"] - ia["end"]
            if gap < 1:
                continue
            if gap > max_gap_sec * fps:
                break
            dist = np.hypot(*(np.array(ib["start_c"]) - np.array(ia["end_c"])))
            if dist > max_dist_frac * diag:
                continue
            if a in apps and b in apps:
                if float(np.dot(apps[a], apps[b])) < app_th:
                    continue
            ra, rb = find(a), find(b)
            if ra != rb:
                mem = [t for t in tids if find(t) in (ra, rb)]
                spans = sorted((info[t]["start"], info[t]["end"]) for t in mem)
                if all(spans[k][1] < spans[k+1][0] for k in range(len(spans)-1)):
                    parent[rb] = ra

    groups = {}
    for t in tids:
        groups.setdefault(find(t), []).append(t)
    return groups


# ================= TARGET SELECTION =================
def score_groups_by_face(video_path, tracks, groups, ref_emb, meta, samples_per_group=6):
    """飯拍模式：每個 group 抽格認樣，計平均相似度"""
    face_app = get_face_app()
    cap = cv2.VideoCapture(video_path)
    scores = {}
    for gid, tids in groups.items():
        frames = sorted(f for t in tids for f in tracks[t])
        picks = frames[:: max(1, len(frames) // samples_per_group)][:samples_per_group]
        sims = []
        for fi in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            box = next(tracks[t][fi] for t in tids if fi in tracks[t])
            x1, y1, x2, y2 = map(int, box)
            pad = int((y2 - y1) * 0.1)
            crop = frame[max(0, y1-pad):y2, max(0, x1-pad):min(meta["W"], x2+pad)]
            if crop.size == 0:
                continue
            faces = face_app.get(crop)
            if faces:
                f = max(faces, key=lambda x: x.det_score)
                sims.append(float(np.dot(f.normed_embedding, ref_emb)))
        scores[gid] = float(np.mean(sims)) if sims else -1.0
    cap.release()
    return scores


def score_groups_auto(tracks, groups, meta):
    """唔使參考相嘅主角評分：出鏡率 × 置中程度 × 平均大細（C 位偵測）"""
    W, H, total = meta["W"], meta["H"], meta["total"]
    scores = {}
    for gid, tids in groups.items():
        n, cen, area = 0, 0.0, 0.0
        for t in tids:
            for fi, (x1, y1, x2, y2) in tracks[t].items():
                n += 1
                cx = (x1 + x2) / 2
                cen += 1.0 - abs(cx - W/2) / (W/2)          # 越近中線分越高
                area += (x2-x1) * (y2-y1) / (W*H)
        if n == 0:
            scores[gid] = 0.0
            continue
        presence = n / total
        scores[gid] = presence * (0.5 + 0.5 * cen/n) * (0.3 + 0.7 * min(area/n * 20, 1.0))
    return scores


def keyframe_hits(tracks, groups, keyframes):
    """keyframes = [(frame_idx, x, y)] → [(frame_idx, gid)]，撳咗邊個就跟邊個"""
    tid2gid = {t: g for g, ts in groups.items() for t in ts}
    hits = []
    for fi, x, y in keyframes:
        best, best_area = None, 1e18
        for tid, fr in tracks.items():
            if fi in fr:
                x1, y1, x2, y2 = fr[fi]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    area = (x2-x1)*(y2-y1)
                    if area < best_area:
                        best, best_area = tid, area
        if best is not None:
            hits.append((fi, tid2gid[best]))
    return sorted(hits)


# ================= TIMELINE + CONFIDENCE =================
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def build_timeline(tracks, groups, selected_gids, meta, kf_hits=None, overlap_th=0.30):
    """
    合併選中 group(s) 逐格 box；keyframe 修正可以喺時間軸中途切換 group。
    回傳 (boxes 已填補, conf 逐格信心 1/0.5/0)
    """
    total = meta["total"]
    boxes = [None] * total

    if kf_hits:
        segs = []
        for i, (fi, gid) in enumerate(kf_hits):
            start = 0 if i == 0 else fi
            end = total if i == len(kf_hits)-1 else kf_hits[i+1][0]
            segs.append((start, end, [gid]))
    else:
        segs = [(0, total, list(selected_gids))]

    for start, end, gids in segs:
        for gid in gids:
            for tid in groups.get(gid, []):
                for fi, box in tracks[tid].items():
                    if start <= fi < end:
                        boxes[fi] = box

    conf = np.array([1.0 if b is not None else 0.0 for b in boxes])

    # 人物重疊偵測：目標框同其他人框 IoU 過高 → 交叉走位時刻，好易靜雞雞跟錯人
    # 就算有偵測都要降信心，逼佢入覆核清單（原作者講嘅「동선 겹침」問題）
    for fi, b in enumerate(boxes):
        if b is None:
            continue
        for tid, fr in tracks.items():
            ob = fr.get(fi)
            if ob is None or ob == b:
                continue
            if _iou(b, ob) > overlap_th:
                conf[fi] = min(conf[fi], 0.6)
                break

    filled = list(boxes)
    i = 0
    while i < total:
        if filled[i] is None:
            j = i
            while j < total and filled[j] is None:
                j += 1
            prev = filled[i-1] if i > 0 else None
            nxt = filled[j] if j < total else None
            if prev is None and nxt is None:
                w, h = meta["W"], meta["H"]
                prev = nxt = (w*0.25, h*0.1, w*0.75, h*0.9)
            elif prev is None:
                prev = nxt
            elif nxt is None:
                nxt = prev
            gap = j - i
            for k in range(i, j):
                if gap <= meta["fps"] * 2:
                    a = (k - i + 1) / (gap + 1)
                    filled[k] = tuple(np.array(prev)*(1-a) + np.array(nxt)*a)
                    conf[k] = 0.5
                else:
                    filled[k] = prev
                    conf[k] = 0.0
            i = j
        else:
            i += 1
    return filled, conf


def review_report(conf, meta, min_len_sec=0.3):
    """低信心時段 [(start_sec, end_sec, mean_conf)] — 要人手覆核嘅位"""
    fps = meta["fps"]
    out, i, n = [], 0, len(conf)
    while i < n:
        if conf[i] < 0.9:
            j = i
            while j < n and conf[j] < 0.9:
                j += 1
            if (j - i) / fps >= min_len_sec:
                out.append((round(i/fps, 2), round(j/fps, 2), round(float(conf[i:j].mean()), 2)))
            i = j
        else:
            i += 1
    return out


def readiness(conf):
    """準備度% = 有實際偵測支持嘅 frame 比例"""
    return round(100 * float((np.asarray(conf) >= 0.9).mean()), 1)


def face_guard(video_path, boxes, conf, tracks, groups, ref_emb, meta,
               check_every_sec=1.0, sim_th=0.30, progress_cb=None):
    """
    沿住已建好嘅 timeline 定期用面容驗證身份（重疊時段後加密抽查）。
    發現跟錯人而畫面有另一個相似度更高嘅框 → 產生自動修正 keyframe。
    回傳 (auto_kf_hits, report)
    """
    face_app = get_face_app()
    fps, W = meta["fps"], meta["W"]
    step = max(int(fps * check_every_sec), 1)
    checkpoints = set(range(0, len(boxes), step))
    # 重疊/低信心段結束後即刻加抽查點（跟錯人最常發生喺呢啲位之後）
    for i in range(1, len(conf)):
        if conf[i-1] < 0.9 <= conf[i]:
            checkpoints.add(i)
    tid2gid = {t: g for g, ts in groups.items() for t in ts}

    cap = cv2.VideoCapture(video_path)
    auto_kf, report = [], []
    for n_done, fi in enumerate(sorted(checkpoints)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or boxes[fi] is None:
            continue

        def face_sim(box):
            x1, y1, x2, y2 = map(int, box)
            pad = int((y2 - y1) * 0.1)
            crop = frame[max(0, y1-pad):y2, max(0, x1-pad):min(W, x2+pad)]
            if crop.size == 0:
                return None
            faces = face_app.get(crop)
            if not faces:
                return None
            f = max(faces, key=lambda x: x.det_score)
            return float(np.dot(f.normed_embedding, ref_emb))

        cur = face_sim(boxes[fi])
        if cur is None:
            continue  # 背向/側面驗唔到，唔亂改
        if cur >= sim_th:
            continue  # 身份正確
        # 目標唔似 → 搵吓其他框有冇更似嘅
        best_tid, best_sim = None, cur
        for tid, fr in tracks.items():
            ob = fr.get(fi)
            if ob is None or ob == boxes[fi]:
                continue
            s = face_sim(ob)
            if s is not None and s > best_sim and s >= sim_th:
                best_tid, best_sim = tid, s
        if best_tid is not None:
            x1, y1, x2, y2 = tracks[best_tid][fi]
            auto_kf.append((fi, (x1+x2)/2, (y1+y2)/2))
            report.append(f"{fi/fps:.1f}s：跟錯人（相似度 {cur:.2f}）→ 自動切去 G{tid2gid[best_tid]}（{best_sim:.2f}）")
        else:
            report.append(f"{fi/fps:.1f}s：目標面容唔匹配（{cur:.2f}）但搵唔到更似嘅人 — 建議人手覆核")
        if progress_cb:
            progress_cb(n_done / max(len(checkpoints), 1), f"Face-Guard 驗證 {n_done}/{len(checkpoints)}")
    cap.release()
    return auto_kf, report


# ================= RENDER =================
def _ema(series, alpha):
    out = np.empty(len(series), dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
    return out


def render(video_path, boxes, meta, out_path,
           aspect_w=9, aspect_h=16, out_height=1920,
           zoom=1.0, smooth_alpha=0.10, headroom=0.12,
           progress_cb=None, preview_cb=None):
    """任意比例 aspect_w:aspect_h、任意輸出高度（闊度自動計，2 對齊）"""
    W, H, fps = meta["W"], meta["H"], meta["fps"]
    ar = aspect_w / aspect_h

    cs = np.array([_center(b) for b in boxes])
    hs = np.array([b[3] - b[1] for b in boxes])
    cx = _ema(cs[:, 0], smooth_alpha)
    cy = _ema(cs[:, 1], smooth_alpha)
    ch = _ema(hs, smooth_alpha)

    crop_h = np.clip(ch * 1.35 * zoom, H * 0.25, H)
    crop_w = crop_h * ar
    over = crop_w > W
    crop_w[over] = W
    crop_h[over] = W / ar

    out_h = int(round(out_height / 2) * 2)
    out_w = int(round(out_h * ar / 2) * 2)

    tmp = out_path + ".video.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    cap = cv2.VideoCapture(video_path)
    i, n = 0, len(boxes)
    while True:
        ok, frame = cap.read()
        if not ok or i >= n:
            break
        w2, h2 = crop_w[i]/2, crop_h[i]/2
        x = float(np.clip(cx[i], w2, W - w2))
        y = float(np.clip(cy[i] - h2*headroom, h2, H - h2))
        crop = frame[int(y-h2):int(y+h2), int(x-w2):int(x+w2)]
        out_frame = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        vw.write(out_frame)
        if preview_cb and i % max(int(fps), 1) == 0:  # 每秒一張實時預覽
            preview_cb(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), i / n)
        i += 1
        if progress_cb and i % 30 == 0:
            progress_cb(i / n, f"輸出中 {i}/{n}")
    cap.release()
    vw.release()

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", video_path,
         "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "medium", "-crf", "17",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path],
        check=True)
    os.remove(tmp)
    return out_path


def render_multi(video_path, named_boxes, meta, out_dir,
                 aspect_w=9, aspect_h=16, out_height=1080,
                 zoom=1.0, smooth_alpha=0.10, headroom=0.12,
                 pose_overlay=False, progress_cb=None):
    """
    全員模式：named_boxes = {名: boxes}，一次 decode 同時出 N 條片。
    pose_overlay=True 會用 YOLOv8-pose 喺每條 crop 畫骨架。
    回傳 [輸出路徑]
    """
    W, H, fps = meta["W"], meta["H"], meta["fps"]
    ar = aspect_w / aspect_h
    out_h = int(round(out_height / 2) * 2)
    out_w = int(round(out_h * ar / 2) * 2)

    plans = {}
    for name, boxes in named_boxes.items():
        cs = np.array([_center(b) for b in boxes])
        hs = np.array([b[3] - b[1] for b in boxes])
        cx = _ema(cs[:, 0], smooth_alpha)
        cy = _ema(cs[:, 1], smooth_alpha)
        ch = _ema(hs, smooth_alpha)
        crop_h = np.clip(ch * 1.35 * zoom, H * 0.25, H)
        crop_w = crop_h * ar
        over = crop_w > W
        crop_w[over] = W
        crop_h[over] = W / ar
        plans[name] = (cx, cy, crop_w, crop_h)

    pose_model = YOLO("yolov8n-pose.pt") if pose_overlay else None
    SKEL = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]

    os.makedirs(out_dir, exist_ok=True)
    writers, tmps = {}, {}
    for name in plans:
        tmp = os.path.join(out_dir, f"{name}.video.mp4")
        tmps[name] = tmp
        writers[name] = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                                        fps, (out_w, out_h))

    cap = cv2.VideoCapture(video_path)
    n = min(len(b) for b in named_boxes.values())
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= n:
            break
        for name, (cx, cy, cw, chh) in plans.items():
            w2, h2 = cw[i]/2, chh[i]/2
            x = float(np.clip(cx[i], w2, W - w2))
            y = float(np.clip(cy[i] - h2*headroom, h2, H - h2))
            crop = frame[int(y-h2):int(y+h2), int(x-w2):int(x+w2)]
            out = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            if pose_model is not None:
                pr = pose_model(out, verbose=False, conf=0.4)[0]
                if pr.keypoints is not None and len(pr.keypoints) > 0:
                    kps = pr.keypoints.xy[0].cpu().numpy()
                    kconf = pr.keypoints.conf[0].cpu().numpy() if pr.keypoints.conf is not None else np.ones(len(kps))
                    for a, b in SKEL:
                        if a < len(kps) and b < len(kps) and kconf[a] > 0.5 and kconf[b] > 0.5:
                            cv2.line(out, tuple(kps[a].astype(int)), tuple(kps[b].astype(int)),
                                     (80, 255, 160), 2)
                    for k, kc in zip(kps, kconf):
                        if kc > 0.5:
                            cv2.circle(out, tuple(k.astype(int)), 3, (255, 200, 60), -1)
            writers[name].write(out)
        i += 1
        if progress_cb and i % 30 == 0:
            progress_cb(i / n, f"全員輸出 {i}/{n}（{len(plans)} 人同步）")

    cap.release()
    outs = []
    for name, vw in writers.items():
        vw.release()
        final = os.path.join(out_dir, f"{name}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", tmps[name], "-i", video_path,
             "-map", "0:v", "-map", "1:a?",
             "-c:v", "libx264", "-preset", "medium", "-crf", "17",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final],
            check=True)
        os.remove(tmps[name])
        outs.append(final)
    return outs


# ================= session save/load =================
def save_session(path, tracks, apps, groups, meta):
    data = {
        "tracks": {str(t): {str(f): list(b) for f, b in fr.items()} for t, fr in tracks.items()},
        "apps": {str(t): v.tolist() for t, v in apps.items()},
        "groups": {str(g): ts for g, ts in groups.items()},
        "meta": meta,
    }
    with open(path, "w") as f:
        json.dump(data, f)

def load_session(path):
    with open(path) as f:
        d = json.load(f)
    tracks = {int(t): {int(f): tuple(b) for f, b in fr.items()} for t, fr in d["tracks"].items()}
    apps = {int(t): np.array(v) for t, v in d["apps"].items()}
    groups = {int(g): ts for g, ts in d["groups"].items()}
    return tracks, apps, groups, d["meta"]


def grab_frame(video_path, frame_idx, tracks=None, groups=None, highlight_gid=None):
    """攞一格 + 畫 group boxes（覆核用）"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if tracks and groups:
        tid2gid = {t: g for g, ts in groups.items() for t in ts}
        for tid, fr in tracks.items():
            if frame_idx in fr:
                x1, y1, x2, y2 = map(int, fr[frame_idx])
                gid = tid2gid.get(tid)
                col = (60, 220, 255) if gid != highlight_gid else (80, 255, 120)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"G{gid}", (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
