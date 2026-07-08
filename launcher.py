"""打包版入口：起 Gradio server + 自動開瀏覽器"""
import threading
import time
import webbrowser


def open_browser():
    time.sleep(2.5)
    webbrowser.open("http://127.0.0.1:7860")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    from app import demo
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, show_error=True)
