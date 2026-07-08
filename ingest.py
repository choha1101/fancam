"""
格式 ingest：
- 影片：任何 ffmpeg 解到嘅格式（webm/mkv/mov/hevc/av1/mts/m2ts/mxf/prores...）
  → 自動轉高質 H.264 proxy 俾 OpenCV 用
- 相：普通格式 cv2 直讀；相機 RAW (NEF/CR3/ARW/DNG...) 用 rawpy 解
"""
import os
import subprocess
import tempfile
import cv2
import numpy as np

RAW_PHOTO_EXTS = {".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2",
                  ".raf", ".rw2", ".orf", ".dng", ".pef", ".x3f"}
RAW_VIDEO_EXTS = {".nev", ".crm", ".braw", ".r3d"}  # 冇開源解碼器


def load_image(path):
    """讀相：RAW 用 rawpy，其餘 cv2。回傳 BGR ndarray or None"""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_PHOTO_EXTS:
        import rawpy
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.imread(path)


def _opencv_readable(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return ok and frame is not None and fps and fps > 0


def prepare_video(path, progress_cb=None):
    """
    回傳 (可用影片路徑, 訊息)。OpenCV 直讀到就原封不動；
    讀唔到但 ffmpeg 解到 → 轉 proxy；RAW 影片 → 明確報錯教點轉。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_VIDEO_EXTS:
        raise ValueError(
            f"{ext.upper()} 係相機 RAW 影片，冇開源解碼器。"
            "請先用 Nikon NX Studio / Canon 官方軟件 / DaVinci Resolve "
            "轉出 ProRes 或 H.265 (MOV/MP4)，再上載。")

    if _opencv_readable(path):
        return path, "直接讀取"

    # ffmpeg 轉高質 proxy（CRF 14 視覺無損級，音軌保留）
    if progress_cb:
        progress_cb(0.0, f"轉換 {ext} → 工作格式…")
    proxy = os.path.join(tempfile.gettempdir(),
                         os.path.splitext(os.path.basename(path))[0] + "_proxy.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", path,
         "-c:v", "libx264", "-preset", "fast", "-crf", "14",
         "-pix_fmt", "yuv420p", "-c:a", "aac", proxy],
        capture_output=True, text=True)
    if r.returncode != 0 or not _opencv_readable(proxy):
        raise ValueError(f"呢個檔案解唔到（{ext}）。ffmpeg 錯誤：{r.stderr[-300:]}")
    return proxy, f"已自動由 {ext} 轉做工作格式"
