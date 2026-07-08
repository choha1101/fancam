"""單元 + smoke tests：唔使真實舞台片都測到成條 pipeline"""
import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fancam_core as fc

META = {"W": 1280, "H": 720, "fps": 30.0, "total": 90}


def make_tracks():
    # G? : tid 1 (frame 0-39) 同 tid 2 (frame 45-89) 係同一個人（位置接近、外觀相似）
    t1 = {f: (100 + f*2, 200, 200 + f*2, 500) for f in range(0, 40)}
    t2 = {f: (185 + (f-45)*2, 200, 285 + (f-45)*2, 500) for f in range(45, 90)}
    t3 = {f: (900, 100, 1000, 600) for f in range(0, 90)}  # 另一個人，全程重疊
    tracks = {1: t1, 2: t2, 3: t3}
    v = np.ones(256)
    v /= np.linalg.norm(v)
    u = np.zeros(256)
    u[0] = 1.0
    apps = {1: v, 2: v, 3: u}
    return tracks, apps


def test_stitch_merges_same_person():
    tracks, apps = make_tracks()
    groups = fc.stitch_tracklets(tracks, apps, META)
    merged = [g for g in groups.values() if set(g) == {1, 2}]
    assert merged, f"tid 1+2 應該縫合埋: {groups}"
    assert any(g == [3] for g in groups.values()), "tid 3 唔應該俾人縫埋"


def test_timeline_and_readiness():
    tracks, apps = make_tracks()
    groups = fc.stitch_tracklets(tracks, apps, META)
    gid = next(g for g, ts in groups.items() if 1 in ts)
    boxes, conf = fc.build_timeline(tracks, groups, [gid], META)
    assert len(boxes) == META["total"] and all(b is not None for b in boxes)
    r = fc.readiness(conf)
    assert 90 <= r < 100          # 85/90 格有偵測 ≈ 94.4%
    gaps = fc.review_report(conf, META, min_len_sec=0.1)
    assert gaps and abs(gaps[0][0] - 40/30) < 0.1  # 報告要指出 40-44 格個 gap


def test_keyframe_hits_switch_target():
    tracks, apps = make_tracks()
    groups = fc.stitch_tracklets(tracks, apps, META)
    hits = fc.keyframe_hits(tracks, groups, [(10, 150, 350), (60, 950, 300)])
    assert len(hits) == 2 and hits[0][1] != hits[1][1]
    boxes, conf = fc.build_timeline(tracks, groups, [], META, kf_hits=hits)
    assert boxes[80][0] > 800     # 後段應該跟緊右邊嗰個


def test_render_custom_aspect_resolution(tmp_path):
    # 合成一條測試片（唔使版權片）
    src = str(tmp_path / "src.mp4")
    vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for f in range(90):
        img = np.zeros((720, 1280, 3), np.uint8)
        cv2.rectangle(img, (100 + f*2, 200), (200 + f*2, 500), (0, 255, 0), -1)
        vw.write(img)
    vw.release()

    tracks, apps = make_tracks()
    groups = fc.stitch_tracklets(tracks, apps, META)
    gid = next(g for g, ts in groups.items() if 1 in ts)
    boxes, _ = fc.build_timeline(tracks, groups, [gid], META)
    out = str(tmp_path / "out.mp4")
    fc.render(src, boxes, META, out, aspect_w=4, aspect_h=5, out_height=480)
    cap = cv2.VideoCapture(out)
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    assert (w, h) == (384, 480)   # 4:5 @ 480 高


@pytest.mark.skipif(os.environ.get("SKIP_YOLO") == "1", reason="skip heavy model")
def test_yolo_detects_people(tmp_path):
    """真實偵測 smoke test：用 ultralytics 官方 sample 相（有人喺入面）"""
    from ultralytics import YOLO
    import urllib.request
    p = str(tmp_path / "bus.jpg")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg", p)
    r = YOLO("yolov8n.pt")(p, classes=[0], verbose=False)[0]
    assert len(r.boxes) >= 3      # 相入面至少 3 個人
