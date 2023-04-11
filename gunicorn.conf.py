bind = '0.0.0.0:80'
pidfile = "api.pid"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"

accesslog = "./logs/accesslog.log"
errorlog = "./logs/errorlog.log"
loglevel = "info"
capture_output = True
