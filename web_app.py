import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import config
from translate import Translator

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_REQUEST_BYTES = 16_384
MAX_SENTENCE_LENGTH = 1_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class TranslationHTTPServer(HTTPServer):
    def __init__(self, server_address, translator):
        self.translator = translator
        super().__init__(server_address, TranslationHandler)


class TranslationHandler(BaseHTTPRequestHandler):
    server: TranslationHTTPServer

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            self._send_json(200, {
                "ready": True,
                "device": str(config.device),
                "model": "Transformer + Beam Search"
            })
            return

        if self.path not in {"/", "/index.html"}:
            self.send_error(404, "Not found")
            return

        page_path = WEB_DIR / "index.html"
        try:
            body = page_path.read_bytes()
        except OSError:
            logging.exception("Could not read the web interface at %s", page_path)
            self.send_error(500, "Could not load the web interface")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/translate":
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "请求格式无效。"})
            return

        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "请求为空或内容过大。"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "请求不是有效的 JSON。"})
            return

        if not isinstance(payload, dict) or not isinstance(payload.get("sentence"), str):
            self._send_json(400, {"error": "请提供英文句子。"})
            return

        sentence = payload["sentence"].strip()
        if not sentence:
            self._send_json(400, {"error": "请输入要翻译的英文句子。"})
            return
        if len(sentence) > MAX_SENTENCE_LENGTH:
            self._send_json(400, {"error": f"句子不能超过 {MAX_SENTENCE_LENGTH} 个字符。"})
            return

        try:
            translation = self.server.translator.translate(sentence)
        except Exception:
            logging.exception("Translation failed")
            self._send_json(500, {"error": "翻译失败，请检查模型文件和运行设备配置。"})
            return

        self._send_json(200, {"translation": translation})

    def log_message(self, format_string, *args):
        logging.info("%s - %s", self.address_string(), format_string % args)


def main():
    translator = Translator()
    server = TranslationHTTPServer(("127.0.0.1", 8000), translator)
    logging.info("Translation demo is ready at http://127.0.0.1:8000 (device: %s)", config.device)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down translation demo")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
