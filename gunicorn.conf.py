from config import Config

bind = "0.0.0.0:5000"
workers = Config.WORKERS
threads = Config.THREADS
worker_class = Config.WORKER_CLASS
timeout = Config.TIMEOUT
max_requests = Config.MAX_REQUESTS
max_requests_jitter = 50
preload_app = True

def on_starting(server):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
