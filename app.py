"""
Fancam Studio v2 — UI
Step 1 分析全片（一次過，之後可以任意改設定重出，唔使再追蹤）
Step 2 揀目標：飯拍=參考相自動認人 / 運動=喺畫面撳個人（keyframe）
Step 3 睇準備度% + 覆核低信心時段，撳圖修正
Step 4 任意比例、任意解像度輸出
"""
import os
import tempfile
import queue
import threading
import gradio as gr
import fancam_core as fc

S = {}  # session: tracks/apps/groups/meta/video/keyframes


def analyze(video, model_size, speed, progress=gr.Progress()):
    if video is None:
        raise gr.Error("請先上載影片")
    video = video.name if hasattr(video, "name") else video
    from ingest import prepare_video
    try:
        video, ingest_msg = prepare_video(video, progress_cb=lambda p, m: progress(p * 0.05, desc=m))
    except ValueError as e:
        raise gr.Error(str(e))
    model = {"快 (n)": "yolov8n.pt", "平衡 (m)": "yolov8m.pt", "最準 (x)": "yolov8x.pt"}[model_size]
    speed_map = {"🚀 極速+（proxy 720p, stride 3, 無GMC）": (3, 640, True, True),
                 "極速（stride 3, 640px）": (3, 640, False, False),
                 "快（stride 2, 960px）": (2, 960, False, False),
                 "最準（逐格, 1280px）": (1, 1280, False, False)}
    stride, imgsz, use_proxy, fast_tracker = speed_map[speed]
    fc._YOLO = None
    fc.get_yolo(model)

    def cb(p, msg):
        progress(p * 0.9, desc=msg)

    track_src, scale = (video, 1.0)
    if use_proxy:
        progress(0.02, desc="製作追蹤 proxy…")
        track_src, scale = fc.make_tracking_proxy(video)
    tracks, apps, meta = fc.track_video(
        track_src, model_name=model, progress_cb=cb, stride=stride, imgsz=imgsz,
        tracker_cfg=fc.TRACKER_CFG_FAST if fast_tracker else None)
    tracks = fc.scale_tracks(tracks, scale)
    if scale != 1.0:
        import cv2
        cap0 = cv2.VideoCapture(video)
        meta["W"] = int(cap0.get(3))
        meta["H"] = int(cap0.get(4))
        cap0.release()
    groups = fc.stitch_tracklets(tracks, apps, meta)
    S.update(video=video, tracks=tracks, apps=apps, groups=groups, meta=meta, keyframes=[])
    img = fc.grab_frame(video, 0, tracks, groups)
    gids = sorted(groups)
    return (img,
            gr.update(choices=[f"G{g}" for g in gids], value=None),
            f"✅ 分析完成：{meta['total']} 格 / {len(tracks)} 條 tracklet → 縫合成 {len(groups)} 個人物。"
            f"下一步：揀目標（飯拍上載參考相自動認 / 或直接喺圖上撳個人）")


def show_frame(t_sec):
    if "meta" not in S:
        return None
    fi = int(t_sec * S["meta"]["fps"])
    fi = min(fi, S["meta"]["total"] - 1)
    S["cur_frame"] = fi
    return fc.grab_frame(S["video"], fi, S["tracks"], S["groups"], S.get("target_gid"))


def click_target(t_sec, evt: gr.SelectData):
    """喺圖上撳個人 → 加 keyframe（運動模式主力 / 覆核修正）"""
    if "meta" not in S:
        return None, "⚠️ 先撳「分析全片」"
    fi = int(t_sec * S["meta"]["fps"])
    kf = (fi, evt.index[0], evt.index[1])
    hit = fc.keyframe_hits(S["tracks"], S["groups"], [kf])
    if not hit:
        return (fc.grab_frame(S["video"], fi, S["tracks"], S["groups"], S.get("target_gid")),
                f"⚠️ {fi/S['meta']['fps']:.1f}s 呢一下撳唔中任何人物框 — 撳返框入面，"
                "或者拉去人物清晰嘅時間點再撳")
    S["keyframes"].append(kf)
    S["kf_hits"] = fc.keyframe_hits(S["tracks"], S["groups"], S["keyframes"])
    S["target_gid"] = S["kf_hits"][-1][1]
    return (fc.grab_frame(S["video"], fi, S["tracks"], S["groups"], S["target_gid"]),
            f"📍 已加 keyframe @ {fi/S['meta']['fps']:.1f}s（共 {len(S['keyframes'])} 個）→ 目標 G{S['target_gid']}")


def auto_lock():
    if "groups" not in S:
        raise gr.Error("先撳「分析全片」")
    scores = fc.score_groups_auto(S["tracks"], S["groups"], S["meta"])
    best = max(scores, key=scores.get)
    S["target_gid"] = best
    S["kf_hits"] = None
    rank = ", ".join(f"G{g}:{s:.2f}" for g, s in sorted(scores.items(), key=lambda x: -x[1])[:5])
    return (f"⭐ 自動鎖定 G{best}（主角評分：{rank}）。如果佢揀錯咗，"
            "直接喺畫面撳返你想跟嗰個人就會覆蓋", f"G{best}")


def auto_face(ref_photos):
    if not ref_photos:
        raise gr.Error("上載 1–3 張目標成員參考相")
    ref = fc.build_reference_embedding([p.name if hasattr(p, "name") else p for p in ref_photos])
    if ref is None:
        raise gr.Error("參考相認唔到樣，換張正面清晰啲")
    scores = fc.score_groups_by_face(S["video"], S["tracks"], S["groups"], ref, S["meta"])
    best = max(scores, key=scores.get)
    S["target_gid"] = best
    S["kf_hits"] = None
    rank = ", ".join(f"G{g}:{s:.2f}" for g, s in sorted(scores.items(), key=lambda x: -x[1])[:5])
    return f"🎯 面容識別鎖定 G{best}（相似度排名：{rank}）", f"G{best}"


def pick_group(label):
    if label:
        S["target_gid"] = int(label.strip("G"))
        S["kf_hits"] = None
    return f"🎯 手動揀咗 G{S['target_gid']}"


def check_readiness():
    if S.get("target_gid") is None and not S.get("kf_hits"):
        raise gr.Error("未揀目標")
    boxes, conf = fc.build_timeline(S["tracks"], S["groups"], [S.get("target_gid")],
                                    S["meta"], S.get("kf_hits"))
    S["boxes"], S["conf"] = boxes, conf
    r = fc.readiness(conf)
    gaps = fc.review_report(conf, S["meta"])
    if not gaps:
        msg = f"## 準備度 {r}% ✅\n全程有偵測支持，可以直接輸出。"
    else:
        rows = "\n".join(f"- {a}s → {b}s（信心 {c}）— 拉 slider 去呢度，撳返目標修正" for a, b, c in gaps)
        msg = f"## 準備度 {r}%\n以下時段要覆核（拉 slider → 喺圖上撳目標人物 → 再撳「計算準備度」）：\n{rows}"
    return msg


def export_all(aspect_preset, custom_w, custom_h, res_preset, custom_res,
               zoom, smooth, pose_overlay, min_presence, progress=gr.Progress()):
    """全員模式：每個出鏡率夠嘅 group 出一條獨立飯拍"""
    if "groups" not in S:
        raise gr.Error("先撳「分析全片」")
    presets = {"9:16 直向": (9, 16), "16:9 橫向": (16, 9), "4:5 IG": (4, 5),
               "1:1 方形": (1, 1), "3:4": (3, 4), "自訂": None}
    ar = presets[aspect_preset] or (int(custom_w), int(custom_h))
    res = {"720p": 720, "1080p": 1080, "1440p": 1440, "4K (2160)": 2160,
           "自訂": int(custom_res)}[res_preset]
    total = S["meta"]["total"]
    named = {}
    for gid, tids in S["groups"].items():
        n_det = len({f for t in tids for f in S["tracks"][t]})
        if n_det / total * 100 >= min_presence:
            boxes, _ = fc.build_timeline(S["tracks"], S["groups"], [gid], S["meta"])
            named[f"member_G{gid}"] = boxes
    if not named:
        raise gr.Error(f"冇 group 出鏡率夠 {min_presence}%，調低門檻再試")

    def cb(p, msg):
        progress(p, desc=msg)

    outs = fc.render_multi(S["video"], named, S["meta"], os.path.join(tempfile.gettempdir(), "fancam_all"),
                           aspect_w=ar[0], aspect_h=ar[1], out_height=res,
                           zoom=zoom, smooth_alpha=smooth,
                           pose_overlay=pose_overlay, progress_cb=cb)
    return outs


def run_face_guard(ref_photos, progress=gr.Progress()):
    if "boxes" not in S:
        raise gr.Error("先撳「計算準備度」")
    if not ref_photos:
        raise gr.Error("Face-Guard 要 1–3 張參考相先驗證到身份")
    ref = fc.build_reference_embedding([p.name if hasattr(p, "name") else p for p in ref_photos])
    if ref is None:
        raise gr.Error("參考相認唔到樣")

    def cb(p, msg):
        progress(p, desc=msg)

    auto_kf, report = fc.face_guard(S["video"], S["boxes"], S["conf"],
                                    S["tracks"], S["groups"], ref, S["meta"], progress_cb=cb)
    if auto_kf:
        S["keyframes"].extend(auto_kf)
        S["kf_hits"] = fc.keyframe_hits(S["tracks"], S["groups"], S["keyframes"])
        boxes, conf = fc.build_timeline(S["tracks"], S["groups"],
                                        [S.get("target_gid")], S["meta"], S["kf_hits"])
        S["boxes"], S["conf"] = boxes, conf
    r = fc.readiness(S["conf"])
    lines = "\n".join(f"- {x}" for x in report) if report else "- 全部抽查點身份正確 ✅"
    return f"## 🛡️ Face-Guard 完成 → 準備度 {r}%\n自動修正 {len(auto_kf)} 處：\n{lines}"


def export(aspect_preset, custom_w, custom_h, res_preset, custom_res,
           zoom, smooth, progress=gr.Progress()):
    if "boxes" not in S:
        raise gr.Error("先撳「計算準備度」")
    presets = {"9:16 直向": (9, 16), "16:9 橫向": (16, 9), "4:5 IG": (4, 5),
               "1:1 方形": (1, 1), "3:4": (3, 4), "自訂": None}
    ar = presets[aspect_preset] or (int(custom_w), int(custom_h))
    res = {"720p": 720, "1080p": 1080, "1440p": 1440, "4K (2160)": 2160,
           "自訂": int(custom_res)}[res_preset]

    def cb(p, msg):
        progress(p, desc=msg)

    out = os.path.join(tempfile.gettempdir(), "fancam_out.mp4")
    q = queue.Queue()
    done = {}

    def preview_cb(img, frac):
        q.put(img)

    def worker():
        try:
            fc.render(S["video"], S["boxes"], S["meta"], out,
                      aspect_w=ar[0], aspect_h=ar[1], out_height=res,
                      zoom=zoom, smooth_alpha=smooth, progress_cb=cb,
                      preview_cb=preview_cb)
        except Exception as e:
            done["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        img = q.get()
        if img is None:
            break
        yield img, None            # 實時預覽逐格更新
    if "err" in done:
        raise gr.Error(str(done["err"]))
    yield None, out                # 完成：出最終影片


with gr.Blocks(title="Fancam Studio v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Fancam Studio v2\n飯拍 + 運動追蹤 · 準備度評分 · 覆核修正 · 任意比例/解像度")
    with gr.Row():
        with gr.Column(scale=3):
            video_in = gr.File(label="原片（mp4/mov/webm/mkv/hevc/mxf/prores… 自動轉換）", file_types=["video", ".webm", ".mkv", ".mts", ".m2ts", ".mxf", ".ts"])
            with gr.Row():
                model_size = gr.Radio(["快 (n)", "平衡 (m)", "最準 (x)"], value="平衡 (m)", label="偵測模型")
            speed = gr.Radio(["🚀 極速+（proxy 720p, stride 3, 無GMC）", "極速（stride 3, 640px）",
                              "快（stride 2, 960px）", "最準（逐格, 1280px）"],
                             value="快（stride 2, 960px）", label="速度模式")
            analyze_btn = gr.Button("1️⃣ 分析全片", variant="primary")
            status = gr.Markdown()
            frame_view = gr.Image(label="畫面（撳人物 = 指定目標 / 加修正 keyframe）", interactive=False)
            t_slider = gr.Slider(0, 600, value=0, step=0.2, label="時間軸（秒）")
            auto_btn = gr.Button("⭐ 自動鎖定主角（唔使相、唔使撳）", variant="primary")
            with gr.Tab("🎤 飯拍模式（面容自動識別）"):
                ref_photos = gr.File(label="目標成員參考相 1–3 張（JPG/PNG 或相機 RAW：NEF/CR3/ARW/DNG…）", file_count="multiple", file_types=["image", ".nef", ".cr2", ".cr3", ".arw", ".raf", ".rw2", ".orf", ".dng", ".pef"])
                face_btn = gr.Button("2️⃣ 自動認人鎖定")
            with gr.Tab("⚽ 運動 / 手動模式"):
                gr.Markdown("直接喺上面畫面**撳目標人物**（球員背向鏡頭都得）。轉波衫顏色相近跟錯咗，就去嗰秒再撳一下修正。")
                group_pick = gr.Radio(choices=[], label="或者直接揀 Group")
            ready_btn = gr.Button("3️⃣ 計算準備度", variant="secondary")
            guard_btn = gr.Button("🛡️ Face-Guard 自動驗證修正（需參考相）", variant="secondary")
            ready_md = gr.Markdown()
        with gr.Column(scale=2):
            gr.Markdown("### 4️⃣ 輸出設定")
            aspect_preset = gr.Radio(["9:16 直向", "16:9 橫向", "4:5 IG", "1:1 方形", "3:4", "自訂"],
                                     value="9:16 直向", label="比例")
            with gr.Row():
                custom_w = gr.Number(value=9, label="自訂比例 W")
                custom_h = gr.Number(value=16, label="自訂比例 H")
            res_preset = gr.Radio(["720p", "1080p", "1440p", "4K (2160)", "自訂"],
                                  value="1080p", label="解像度（輸出高度，闊度按比例自動計）")
            custom_res = gr.Number(value=1920, label="自訂高度 px")
            zoom = gr.Slider(0.7, 1.8, value=1.0, step=0.05, label="鏡頭距離")
            smooth = gr.Slider(0.03, 0.4, value=0.10, step=0.01, label="跟鏡靈敏度（細=穩）")
            export_btn = gr.Button("✨ 輸出（單人）", variant="primary", size="lg")
            live_preview = gr.Image(label="🔴 實時預覽（輸出緊嘅畫面）", interactive=False)
            out_video = gr.Video(label="完成影片")
            gr.Markdown("### 👥 全員模式")
            min_presence = gr.Slider(10, 90, value=50, step=5,
                                     label="最低出鏡率%（過濾伴舞/路人）")
            pose_overlay = gr.Checkbox(label="加骨架 pose 追蹤（跳舞動作分析用）", value=False)
            export_all_btn = gr.Button("👥 全員一次過輸出", variant="secondary", size="lg")
            out_files = gr.File(label="全員飯拍（每人一條）", file_count="multiple")

    analyze_btn.click(analyze, [video_in, model_size, speed], [frame_view, group_pick, status])
    t_slider.change(show_frame, t_slider, frame_view)
    frame_view.select(click_target, [t_slider], [frame_view, status])
    face_btn.click(auto_face, [ref_photos], [status, group_pick])
    auto_btn.click(auto_lock, None, [status, group_pick])
    group_pick.change(pick_group, group_pick, status)
    ready_btn.click(check_readiness, None, ready_md)
    guard_btn.click(run_face_guard, [ref_photos], ready_md)
    export_btn.click(export, [aspect_preset, custom_w, custom_h, res_preset, custom_res,
                              zoom, smooth], [live_preview, out_video])
    export_all_btn.click(export_all, [aspect_preset, custom_w, custom_h, res_preset, custom_res,
                                      zoom, smooth, pose_overlay, min_presence], out_files)

if __name__ == "__main__":
    demo.launch()
