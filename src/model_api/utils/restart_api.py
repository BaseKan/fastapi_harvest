import subprocess


def restart_api():
    subprocess.Popen(['sh', 'start_api.sh'], start_new_session=True)
