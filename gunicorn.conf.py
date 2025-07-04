# Gunicorn configuration for CodeSandbox
# Optimized for limited resources and high efficiency

# Server socket
bind = "0.0.0.0:5000"
backlog = 128

# Worker processes
workers = 1  # Single worker for CodeSandbox resource limits
worker_class = "sync"
worker_connections = 50  # Reduced for limited memory
max_requests = 100  # Restart workers after 100 requests to prevent memory leaks
max_requests_jitter = 10  # Add randomness to restart timing
preload_app = True  # Load app before forking workers (saves memory)

# Timeouts
timeout = 30  # Request timeout
keepalive = 2  # Keep connections alive for reuse
graceful_timeout = 20

# Logging
loglevel = "info"
capture_output = True
enable_stdio_inheritance = True

# Process naming
proc_name = "football_predictor"

# Resource limits for CodeSandbox
limit_request_line = 2048  # Smaller request lines
limit_request_fields = 50  # Fewer header fields
limit_request_field_size = 4096  # Smaller header size

# Memory management
tmp_upload_dir = "/tmp"  # Use /tmp for uploads

# Error recovery
restart_on_failure = True

def worker_abort(worker):
    """Handle worker aborts gracefully"""
    worker.log.info("Worker aborted, cleaning up...")

def on_starting(server):
    """Called just before the master process is initialized"""
    server.log.info("Football Predictor starting with CodeSandbox optimizations...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP"""
    server.log.info("Reloading workers...")

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal"""
    worker.log.info("Worker interrupted")

def pre_fork(server, worker):
    """Called before forking a worker"""
    server.log.info(f"Worker {worker.age} spawned")

def post_fork(server, worker):
    """Called after a worker has been forked"""
    server.log.info(f"Worker {worker.pid} booted")

def worker_exit(server, worker):
    """Called when a worker is exiting"""
    server.log.info(f"Worker {worker.pid} exiting")