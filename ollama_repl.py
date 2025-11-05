# 프롬프트 수정을 통해 질문/답변이 제대로 이뤄지는지 확인하는 용도

# ollama_repl.py
import os, sys, json, time, argparse, requests
from datetime import datetime
from pathlib import Path

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL      = os.getenv("OLLAMA_MODEL", "eeve-korean-10_8b")

session = requests.Session()
session.trust_env = False  # 프록시 간섭 방지

def load_text(path: str | None) -> str:
    if not path: return ""
    p = Path(path)
    return p.read_text(encoding="utf-8").strip()

def chat_once(
    user_text: str,
    system_text: str = "",
    stream: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    repeat_penalty: float = 1.15,
    num_predict: int = 120,
    stop: list[str] | None = None,
    timeout: int = 120,
) -> str:
    """Ollama /api/chat 1회 호출 (스트리밍/비스트리밍 모두 지원)"""
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "num_predict": num_predict,
        }
    }
    if stop:
        payload["options"]["stop"] = stop

    try:
        if stream:
            with session.post(OLLAMA_URL, json=payload, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                out = []
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # 혹시 공백/잡텍스트 섞이면 무시
                        continue
                    delta = obj.get("message", {}).get("content", "")
                    if delta:
                        print(delta, end="", flush=True)  # 터미널에 스트리밍
                        out.append(delta)
                    if obj.get("done"):
                        break
                print()  # 줄바꿈
                return "".join(out).strip()
        else:
            r = session.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "").strip()
    except requests.RequestException as e:
        return f"[HTTP ERROR] {e}"
    except Exception as e:
        return f"[ERROR] {e}"

def save_log(text: str, log_dir: str = "./logs") -> str:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    path = Path(log_dir) / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)

def run_repl(system_file: str | None, stream: bool, opts: dict):
    system_text = load_text(system_file)
    print("== Ollama REPL ==")
    print(f"model={MODEL} stream={stream} temp={opts['temperature']} top_p={opts['top_p']} repeat_penalty={opts['repeat_penalty']} num_predict={opts['num_predict']}")
    print("명령: /sys <path> (시스템프롬프트 교체), /temp 0.6, /top_p 0.9, /rp 1.15, /np 120, /stream on|off, /save, /exit")

    transcript = []
    while True:
        try:
            user = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            break

        if not user:
            continue

        # 명령 처리
        if user.startswith("/"):
            parts = user.split()
            cmd = parts[0].lower()

            if cmd == "/exit":
                break
            elif cmd == "/sys":
                if len(parts) < 2:
                    print("사용법: /sys path/to/system.txt")
                else:
                    system_file = " ".join(parts[1:])
                    system_text = load_text(system_file)
                    print(f"시스템 프롬프트 갱신 ({len(system_text)} chars)")
            elif cmd == "/temp" and len(parts) == 2:
                opts["temperature"] = float(parts[1]); print("temperature =", opts["temperature"])
            elif cmd == "/top_p" and len(parts) == 2:
                opts["top_p"] = float(parts[1]); print("top_p =", opts["top_p"])
            elif cmd == "/rp" and len(parts) == 2:
                opts["repeat_penalty"] = float(parts[1]); print("repeat_penalty =", opts["repeat_penalty"])
            elif cmd == "/np" and len(parts) == 2:
                opts["num_predict"] = int(parts[1]); print("num_predict =", opts["num_predict"])
            elif cmd == "/stream" and len(parts) == 2:
                stream = (parts[1].lower() == "on"); print("stream =", stream)
            elif cmd == "/save":
                log_path = save_log("\n".join(transcript))
                print("저장:", log_path)
            else:
                print("알 수 없는 명령")
            continue

        # 일반 대화
        print("Eeve >", end=" ", flush=True) if stream else None
        t0 = time.time()
        ans = chat_once(
            user_text=user,
            system_text=system_text,
            stream=stream,
            **opts,
        )
        t1 = time.time()
        if not stream:
            print(f"Eeve > {ans}")
        print(f"[{t1 - t0:.2f}s]")

        transcript.append(f"You: {user}")
        transcript.append(f"Eeve: {ans}")

def main():
    ap = argparse.ArgumentParser(description="Ollama local chat (VSCode test)")
    ap.add_argument("--system", type=str, default=None, help="시스템 프롬프트 파일 경로(.txt)")
    ap.add_argument("--once", type=str, default=None, help="한 번만 질문하고 종료(문자열)")
    ap.add_argument("--stream", action="store_true", help="스트리밍 출력 사용")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repeat_penalty", type=float, default=1.15)
    ap.add_argument("--num_predict", type=int, default=120)
    args = ap.parse_args()

    opts = dict(
        temperature=args.temp,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        num_predict=args.num_predict,
    )

    if args.once:
        sys_prompt = load_text(args.system)
        ans = chat_once(args.once, system_text=sys_prompt, stream=args.stream, **opts)
        print("Eeve >", ans)
    else:
        run_repl(args.system, stream=args.stream, opts=opts)

if __name__ == "__main__":
    main()
