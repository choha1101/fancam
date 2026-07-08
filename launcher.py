"""打包版入口：起 Gradio server + 自動開瀏覽器 + crash 寫 log"""
import multiprocessing
import os
import sys
import threading
import time
import traceback
import webbrowser


def open_browser():
    time.sleep(3)
    webbrowser.open("http://127.0.0.1:7860")


def main():
    threading.Thread(target=open_browser, daemon=True).start()
    from app import demo
    demo.launch(server_name="127.0.0.1", server_port=7860,
                inbrowser=False, show_error=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # PyInstaller 必需，冇佢會即 crash
    try:
        main()
    except Exception:
        # crash 就寫 log 落 app 隔籬，雙擊 error_log.txt 就睇到死因
        log = os.path.join(os.path.dirname(sys.executable), "error_log.txt")
        with open(log, "w") as f:
            f.write(traceback.format_exc())
        raise
