

_occupy_process = None
_stop_event = None
_started = False

def occupy_gpu(gpu_id, size, ready_event, stop_event):
    import paddle
    import time

    paddle.device.set_device(f"gpu:{gpu_id}")

    x = paddle.ones([size, size])

    ready_event.set()

    while not stop_event.is_set():
            x = x * 2 - 1

def warmup_paddle():
    import multiprocessing as mp
    import paddle
    import os
    import atexit
    import time
    global _occupy_process, _stop_event, _started

    # ✅ 如果已经启动过，就直接返回
    if _started and _occupy_process is not None and _occupy_process.is_alive():
        print("GPU already warmed up")
        return

    mp.set_start_method("spawn", force=True)

    gpu_id = paddle.cuda.current_device()
    size = 1

    ready_event = mp.Event()
    _stop_event = mp.Event()

    _occupy_process = mp.Process(
        target=occupy_gpu,
        args=(gpu_id, size, ready_event, _stop_event),
    )
    _occupy_process.start()

    ready_event.wait()
    _started = True

    print("GPU warm up success")

    # 自动清理（只注册一次）
    def cleanup():
        global _occupy_process, _stop_event, _started
        if _occupy_process is not None and _occupy_process.is_alive():
            print("exit!")
            _stop_event.set()
            _occupy_process.join()
        _started = False

    atexit.register(cleanup)