"""
Fancam Studio v2 — UI
Step 1 分析全片（一次過，之後可以任意改設定重出，唔使再追蹤）
Step 2 揀目標：飯拍=參考相自動認人 / 運動=喺畫面撳個人（keyframe）
Step 3 睇準備度% + 覆核低信心時段，撳圖修正
Step 4 任意比例、任意解像度輸出
"""
import gradio as gr
import fancam_core as fc

S = {}  # session: tracks/apps/groups/meta/video/keyframes


def analyze(video, model_size, progress=gr.Progress()):
    if video is None:
        raise gr.Error("請先上載影片")
    model = {"快 (n)": "yolov8n.pt", "平衡 (m)": "yolov8m.pt", "最準 (x)": "yolov8x.pt"}[model_size]
    fc._YOLO = None
    fc.get_yolo(model)

    def cb(p, msg):
        progress(p * 0.9, desc=msg)

    tracks, apps, meta = fc.track_video(video, model_name=model, progress_cb=cb)
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
    fi = int(t_sec * S["meta"]["fps"])
    S["keyframes"].append((fi, evt.index[0], evt.index[1]))
    hits = fc.keyframe_hits(S["tracks"], S["groups"], S["keyframes"])
    if hits:
        S["target_gid"] = hits[-1][1]
        S["kf_hits"] = hits
    return (fc.grab_frame(S["video"], fi, S["tracks"], S["groups"], S.get("target_gid")),
            f"📍 已加 keyframe @ {fi/S['meta']['fps']:.1f}s（共 {len(S['keyframes'])} 個）→ 目標 G{S.get('target_gid','?')}")


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

    out = "/tmp/fancam_out.mp4"
    fc.render(S["video"], S["boxes"], S["meta"], out,
              aspect_w=ar[0], aspect_h=ar[1], out_height=res,
              zoom=zoom, smooth_alpha=smooth, progress_cb=cb)
    return out


with gr.Blocks(title="Fancam Studio v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Fancam Studio v2\n飯拍 + 運動追蹤 · 準備度評分 · 覆核修正 · 任意比例/解像度")
    with gr.Row():
        with gr.Column(scale=3):
            video_in = gr.Video(label="原片")
            with gr.Row():
                model_size = gr.Radio(["快 (n)", "平衡 (m)", "最準 (x)"], value="平衡 (m)", label="偵測模型")
                analyze_btn = gr.Button("1️⃣ 分析全片", variant="primary")
            status = gr.Markdown()
            frame_view = gr.Image(label="畫面（撳人物 = 指定目標 / 加修正 keyframe）", interactive=False)
            t_slider = gr.Slider(0, 600, value=0, step=0.2, label="時間軸（秒）")
            with gr.Tab("🎤 飯拍模式（面容自動識別）"):
                ref_photos = gr.File(label="目標成員參考相 1–3 張", file_count="multiple", file_types=["image"])
                face_btn = gr.Button("2️⃣ 自動認人鎖定")
            with gr.Tab("⚽ 運動 / 手動模式"):
                gr.Markdown("直接喺上面畫面**撳目標人物**（球員背向鏡頭都得）。轉波衫顏色相近跟錯咗，就去嗰秒再撳一下修正。")
                group_pick = gr.Radio(choices=[], label="或者直接揀 Group")
            ready_btn = gr.Button("3️⃣ 計算準備度", variant="secondary")
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
            export_btn = gr.Button("✨ 輸出", variant="primary", size="lg")
            out_video = gr.Video(label="輸出預覽")

    analyze_btn.click(analyze, [video_in, model_size], [frame_view, group_pick, status])
    t_slider.change(show_frame, t_slider, frame_view)
    frame_view.select(click_target, [t_slider], [frame_view, status])
    face_btn.click(auto_face, [ref_photos], [status, group_pick])
    group_pick.change(pick_group, group_pick, status)
    ready_btn.click(check_readiness, None, ready_md)
    export_btn.click(export, [aspect_preset, custom_w, custom_h, res_preset, custom_res,
                              zoom, smooth], out_video)

if __name__ == "__main__":
    demo.launch()
