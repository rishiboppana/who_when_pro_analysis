import shutil
out = "/content/whowhen_qwen_results"
shutil.copy("/content/vllm_server.log", f"{E1}/outputs/vllm_server.log")
try: os.killpg(proc.pid, signal.SIGKILL)       # stop the server
except Exception: pass
shutil.make_archive(out, "zip", PROJECT)       # whole project folder (outputs, tables, figures, logs)
print("results zip:", out + ".zip", f"({os.path.getsize(out + '.zip') / 1e6:.1f} MB)", "| persisted on Drive" if USING_DRIVE else "| NOT on Drive: download it now")
try:
    from google.colab import files
    files.download(out + ".zip")
except Exception as e:
    print("files.download not available here:", type(e).__name__, "- fetch the zip from the file panel.")
