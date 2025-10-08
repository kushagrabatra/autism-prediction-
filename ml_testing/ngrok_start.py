# ngrok_start.py (robust auto-reconnect + auto-open)
import subprocess, time, sys, webbrowser
from pyngrok import ngrok, conf

PORT = 8000
UVICORN_CMD = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(PORT)]

def start_uvicorn():
    print("🔹 Starting uvicorn ...")
    return subprocess.Popen(UVICORN_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def get_tunnel_for_port(port):
    try:
        for t in ngrok.get_tunnels():
            # t is NgrokTunnel object
            cfg = getattr(t, "config", None)
            addr = None
            if cfg and isinstance(cfg, dict):
                addr = cfg.get("addr")
            else:
                # fallback attempt
                addr = getattr(t, "addr", None) or getattr(t, "remote_addr", None)
            if addr and str(port) in str(addr):
                return t
    except Exception:
        return None
    return None

def create_tunnel(port):
    print("🌐 Creating ngrok tunnel...")
    return ngrok.connect(port, "http")

def ensure_tunnel(port):
    existing = get_tunnel_for_port(port)
    if existing:
        print("ℹ️ Reusing existing tunnel:", existing.public_url)
        return existing
    # create new
    return create_tunnel(port)

def main():
    # ensure auth token loaded (optional)
    # conf.get_default().auth_token = "YOUR_TOKEN"  # if you want to set in script (not recommended)
    uvicorn_proc = start_uvicorn()
    time.sleep(1.5)

    try:
        tunnel = ensure_tunnel(PORT)
        public_url = getattr(tunnel, "public_url", None)
        if public_url:
            frontend_url = public_url.rstrip("/") + "/frontend/index.html"
            print("✅ Public URL:", public_url)
            print("🔗 Frontend:", frontend_url)
            # open browser once
            webbrowser.open(frontend_url)
        else:
            print("⚠️ Could not determine public_url. Tunnel object:", tunnel)

        print("Monitoring tunnel. Press Ctrl+C to stop.")
        # monitor loop: if ngrok disconnects, try to reconnect
        while True:
            time.sleep(3)
            t = get_tunnel_for_port(PORT)
            if not t:
                print("⚠️ Tunnel disappeared — attempting to recreate...")
                try:
                    new_t = create_tunnel(PORT)
                    public_url = getattr(new_t, "public_url", None)
                    if public_url:
                        frontend_url = public_url.rstrip("/") + "/frontend/index.html"
                        print("✅ Reconnected. New URL:", frontend_url)
                        webbrowser.open(frontend_url)
                    else:
                        print("⚠️ Created tunnel but no public_url returned.")
                except ngrok_exceptions.PyngrokNgrokError as e:
                    print("❌ Failed to recreate tunnel:", e)
                    time.sleep(2)
            # continue loop
    except KeyboardInterrupt:
        print("\n🛑 Stopping.")
    finally:
        try:
            if uvicorn_proc and uvicorn_proc.poll() is None:
                uvicorn_proc.terminate()
                uvicorn_proc.wait(timeout=5)
        except Exception:
            pass

if __name__ == "__main__":
    main()
