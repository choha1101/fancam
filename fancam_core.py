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
    app = get_face_app()
    embs = []
    for p in image_paths:
        img = cv2.imread(p)
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
def track_video(video_path, model_name="yolov8m.pt", conf=0.3, imgsz=1280,
                progress_cb=None, sample_appearance_every=5):
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
        r = model.track(frame, classes=[0], persist=True, verbose=False,
                        conf=conf, imgsz=imgsz, tracker=TRACKER_CFG)[0]
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
def build_timeline(tracks, groups, selected_gids, meta, kf_hits=None):
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
           progress_cb=None):
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
        vw.write(cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4))
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
