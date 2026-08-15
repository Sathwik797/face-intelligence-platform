from app import create_app
from config import load_config

config = load_config("config/config.yaml")
app = create_app("config/config.yaml")

if __name__ == "__main__":
    host = config.get("app", {}).get("host", "127.0.0.1")
    port = config.get("app", {}).get("port", 5000)
    debug = config.get("app", {}).get("debug", True)
    print(f"[INFO] Starting Face Recognition Attendance System on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)