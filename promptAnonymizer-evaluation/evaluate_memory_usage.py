import requests
import time
import psutil
import threading

models = [
    "llama3.2",
    "llama3:8b",
    "phi3:mini",
    "qwen2.5:3b",
    "mistral:7b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "phi3:14b"
]

def get_ollama_runner_pid():
    max_mem = 0
    pid = -1
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            # Check if it's an ollama runner
            if proc.info['name'] == 'ollama_llama_server' or (proc.info['cmdline'] and 'runner' in proc.info['cmdline'] and 'ollama' in proc.info['cmdline'][0]):
                mem = proc.info['memory_info'].rss
                if mem > max_mem:
                    max_mem = mem
                    pid = proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pid

def monitor_resources(pid, stop_event, stats):
    try:
        proc = psutil.Process(pid)
        proc.cpu_percent(interval=None)
        
        ram_samples = []
        
        while not stop_event.is_set():
            try:
                # CPU - interval acts as the sleep
                cpu = proc.cpu_percent(interval=0.1)
                stats['peak_cpu'] = max(stats.get('peak_cpu', 0), cpu)
                
                # RAM
                mem = proc.memory_info().rss
                ram_samples.append(mem)
                stats['peak_ram'] = max(stats.get('peak_ram', 0), mem)
                
            except:
                break
        
        if ram_samples:
            stats['avg_ram'] = sum(ram_samples) / len(ram_samples)
        else:
            # Fallback if loop didn't run
            try:
                mem = proc.memory_info().rss
                stats['avg_ram'] = mem
                stats['peak_ram'] = mem
            except:
                pass
            
    except:
        pass

print(f"{'Model':<15} | {'Avg RAM (GB)':<12} | {'Peak RAM (GB)':<13} | {'Peak CPU (%)':<12} | {'PID':<10}")
print("-" * 75)

for model in models:
    # 0. Unload everything first
    try:
        requests.post("http://localhost:11434/api/generate", json={"model": model, "keep_alive": 0})
        time.sleep(3)
    except:
        pass

    # 1. Trigger model load (Warmup)
    try:
        requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 1}
        })
    except Exception as e:
        print(f"{model:<15} | Error: {e}")
        continue

    # 2. Wait for it to stabilize
    time.sleep(3)

    # 3. Get PID
    pid = get_ollama_runner_pid()
    
    if pid == -1:
        print(f"{model:<15} | {'N/A':<12} | {'N/A':<13} | {'N/A':<12} | {'N/A':<10}")
        continue

    # 4. Monitor during Inference
    stats = {'peak_cpu': 0, 'peak_ram': 0, 'avg_ram': 0}
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(pid, stop_event, stats))
    monitor_thread.start()
    
    try:
        # Send a longer request to generate load
        requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": "Write a detailed essay about the history of the internet and its impact on modern society.",
            "stream": False,
            "options": {"num_predict": 300}
        })
    except:
        pass
    
    stop_event.set()
    monitor_thread.join()
    
    avg_ram_gb = stats['avg_ram'] / (1024**3)
    peak_ram_gb = stats['peak_ram'] / (1024**3)
    
    print(f"{model:<15} | {avg_ram_gb:<12.2f} | {peak_ram_gb:<13.2f} | {stats['peak_cpu']:<12.1f} | {pid:<10}")

    # 5. Force kill the runner to ensure clean state
    if pid > 0:
        try:
            p = psutil.Process(pid)
            p.kill()
            p.wait(timeout=3)
        except:
            pass
    time.sleep(2)
