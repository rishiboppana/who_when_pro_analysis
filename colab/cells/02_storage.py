import os, sys, json, time, zipfile, subprocess, base64, lzma, tarfile, io, hashlib, shutil, re   # imported here too, so each cell also works on its own
USING_DRIVE = False
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT, USING_DRIVE = "/content/drive/MyDrive/whowhen_colab", True
except Exception as e:
    print(f"Drive mount not available here ({type(e).__name__}: {str(e)[:150]}). Using local /content (download the results at the end).")
    PROJECT = "/content/whowhen_colab"
os.makedirs(PROJECT, exist_ok=True)
E1, E2 = f"{PROJECT}/experiment_1", f"{PROJECT}/experiment_2"
print("USING_DRIVE =", USING_DRIVE, "| PROJECT =", PROJECT)
