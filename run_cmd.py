import os, sys
from subprocess import call

def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)

command = (
        "python upload.py "
        + " --local_file "
        + local_file
        + " --upload_file "
        + upload_file
)
run_cmd(command)

