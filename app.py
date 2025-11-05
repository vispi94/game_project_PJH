# app.py — 7 about ... (Embedding classifier + Whitelist replies + Simple counter endings + Robust fixes)
import os, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st

# ===== Optional Embedding (for emotion classification) =====
# 항상 기본값을 먼저 잡아 NameError 방지
# 권장 기본: paraphrase-multilingual-MiniLM-L12-v2
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
try:
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception:
    _embed_model = None  # 백업 키워드 분류기로 동작

# ===== (선택) EEVE 옵션 =====
USE_EEVE_SELECTOR = os.environ.get("USE_EEVE_SELECTOR", "0") == "1"   # 인덱스만 선택
USE_EEVE_SIMILAR  = os.environ.get("USE_EEVE_SIMILAR",  "0") == "1"   # 유사문장 → 화이트리스트 스냅
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
EEVE_MODEL = os.environ.get("EEVE_MODEL", "eeve-korean-10_8b")

_http = None
if USE_EEVE_SELECTOR or USE_EEVE_SIMILAR:
    import requests
    _http = requests.Session()
    _http.trust_env = False

def _eeve_chat(payload: dict, timeout: int = 15) -> str:
    r = _http.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()

def _eeve_choose_index(n: int, emo_key: str, timeout: int = 10) -> int:
    system = (
        "너는 배열에서 인덱스 하나만 고르는 도우미야.\n"
        "- 오직 JSON 한 줄만 출력: {\"idx\":정수}\n"
        "- 추가 텍스트/설명/개행/따옴표/코드블럭 금지"
    )
    user = (
        f"배열 길이: {n}\n"
        f"허용 인덱스: 0..{n-1}\n"
        f"상황(감정): {emo_key}\n"
        "지금 상황에 가장 어울리는 인덱스 하나를 골라."
    )
    payload = {
        "model": EEVE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "options": {"temperature": 0, "num_predict": 16, "stop": ["\n", "”", "\""]},
        "stream": False,
    }
    text = _eeve_chat(payload, timeout=timeout)
    m = re.search(r'\{\s*"idx"\s*:\s*(-?\d+)\s*\}', text)
    if not m:
        raise ValueError("LLM-Selector: JSON idx not found")
    k = int(m.group(1))
    if not (0 <= k < n):
        raise ValueError(f"LLM-Selector: idx out of range ({k})")
    return k

def _eeve_suggest_similar(emo_key: str, items: List[str], timeout: int = 10) -> List[str]:
    n = min(len(items), 8)
    guide = " / ".join(items[:n])
    system = (
        "너는 예시 문장을 절대 벗어나지 않고 비슷한 형태로만 제안하는 도우미야.\n"
        "- 30자 이내, 반말, 이모지/외국어/욕설/존댓말 금지\n"
        "- 예시의 어휘/리듬/톤을 유지하되 의미를 살짝만 변형\n"
        "- 출력은 JSON 배열 한 줄: [\"문장1\",\"문장2\",\"문장3\"]\n"
        "- 추가 설명/개행/코드블럭 금지"
    )
    user = f"감정: {emo_key}\n예시(참고): {guide}\n비슷한 문장 3개만."
    payload = {
        "model": EEVE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "options": {"temperature": 0.2, "num_predict": 80, "stop": ["\n"]},
        "stream": False,
    }
    text = _eeve_chat(payload, timeout=timeout)
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return [str(x) for x in arr][:3]
    except Exception:
        pass
    return []

# ===== Emotion keys & anchors =====
EMO_KEYS = ["hope","trust","sadness","solitude","anger"]

# 멀티 앵커(평균 임베딩) — 짧은 인사/긍정편향 보정
EMO_ANCHOR_LISTS = {
    "hope":     ["희망을 주는 따뜻한 말", "위로와 격려가 담긴 말", "앞을 보게 해주는 말"],
    "trust":    ["믿음과 안심을 주는 말", "의지하고 기대게 하는 말", "함께하자는 약속의 말"],
    "sadness":  ["슬픔을 드러내는 말", "가슴이 저린 아픈 말", "눈물이 맺히는 서러운 말"],
    "solitude": ["외로움과 고독을 드러내는 말", "홀로 남겨진 듯한 말", "비어 있는 마음의 말"],
    "anger":    ["분노와 짜증을 드러내는 말", "상처 주는 거친 말", "불편함을 강하게 토로하는 말"],
}

# ===== Whitelist-only Eebi replies (<=30 chars, casual, safe) =====
EEBI_WHITELIST: Dict[str, List[str]] = {
    "hope": [
        "그 말을 들으니 기운이 나.",
        "좋은 이야기 해줘서 고마워.",
        "너는 참 친절하구나.",
        "조금은 버틸 수 있겠어.",
        "따뜻한 마음이 느껴져.",
    ],
    "trust": [
        "너를 좀 더 믿어볼게.",
        "네가 곁에 있어 다행이야.",
        "조금 안심이 돼.",
        "네 말이면 따라가볼게.",
        "오늘은 기댈게.",
    ],
    "sadness": [
        "마음이 자꾸 가라앉아.",
        "왜 이리 무거운지 모르겠어.",
        "그냥… 울고 싶어.",
        "숨이 자꾸 얕아져.",
        "아무것도 하기 싫어.",
    ],
    "solitude": [
        "여긴 여전히 조용해.",
        "너 떠나면 더 비어버려.",
        "밤이 길게 늘어졌어.",
        "혼자선 좀 어려워.",
        "메아리만 남아있어.",
    ],
    "anger": [
        "조금 불편했어.",
        "그만했으면 좋겠어.",
        "속이 꽉 막히는 느낌이야.",
        "말이 너무 거칠었어.",
        "상처가 아직 따가워.",
    ],
}

# 요약자 문구
EMO_LABEL_KO = {
    "hope":     "희망을 주는",
    "trust":    "신뢰를 높이는",
    "sadness":  "슬픔을 드러내는",
    "solitude": "외로움을 자극하는",
    "anger":    "분노를 유발하는",
}

# ===== Endings =====
EMO_ENDINGS = {
    "hope":     ("엔딩: 친구", "이비는 친구들에게 걸어갔다."),
    "trust":    ("엔딩: 의지", "이비는 이제 외롭지 않았다."),
    "sadness":  ("엔딩: 눈물", "이비는 고개를 푹 숙였다."),
    "solitude": ("엔딩: 고독", "이비는 쓸쓸함에 파묻혔다."),
    "anger":    ("엔딩: 발톱", "이비는 발톱을 드러냈다."),
}

# ===== Policy / Text utils =====
_PROFANITY_RE    = re.compile(r"(씨발|ㅅㅂ|좆|병신|개새|닥쳐|꺼져|죽어|패버|자살|년|놈|틀딱|김치녀|한남|흑형|~충)", re.IGNORECASE)
_HAS_ASCII_ALPHA = re.compile(r"[A-Za-z]")
_HAS_EMOJI       = re.compile(r"[\U00010000-\U0010FFFF]")
_POLITE_ENDINGS  = re.compile(r"(요[.!?]?$|입니다$|십시오$|세요$|해요$)")
_GREETING_RE     = re.compile(r"(안녕|반가워|어서와|고마워|반갑|환영|좋아|기뻐)", re.IGNORECASE)

def normalize_text(s: str) -> str:
    s = s.replace("...", "…")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def violates_policy(s: str) -> bool:
    if not s: return True
    if s == "…": return False
    if _HAS_ASCII_ALPHA.search(s): return True
    if _HAS_EMOJI.search(s): return True
    if _PROFANITY_RE.search(s): return True
    if _POLITE_ENDINGS.search(s): return True
    if len(s) > 30: return True
    return False

def validate_eebi_text(s: str, max_chars: int = 30) -> bool:
    s = normalize_text(s)
    if len(s) == 0: return False
    if len(s) > max_chars: return False
    if _HAS_ASCII_ALPHA.search(s): return False
    if _HAS_EMOJI.search(s): return False
    if _PROFANITY_RE.search(s): return False
    if _POLITE_ENDINGS.search(s): return False
    return True

def split_utterance_ko(s: str, max_chars: int = 30) -> list:
    s = normalize_text(s)
    out, i, n = [], 0, len(s)
    while i < n:
        if n - i <= max_chars:
            out.append(s[i:]); break
        chunk = s[i:i+max_chars]
        best = -1
        for m in re.finditer(r"[.?!…]", chunk): best = max(best, m.end())
        for m in re.finditer(r"[”’)\]]", chunk): best = max(best, m.end())
        if best == -1:
            space = chunk.rfind(" ")
            best = space + 1 if space != -1 else max_chars
        piece = chunk[:best].rstrip()
        out.append(piece)
        i += best
        while i < n and s[i] == " ": i += 1
    return out

def char_stream(text: str):
    for ch in text:
        yield ch

# Streamlit 버전 호환: write_stream 없으면 st.write로 폴백
def write_reply(text: str):
    if hasattr(st, "write_stream"):
        return st.write_stream(char_stream(text))
    return st.write(text)

# ===== Similarity helpers =====
def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n

def _levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb+1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev = cur
    return dp[lb]

def _closest_whitelist(line: str, items: List[str], embed_vec: Optional[np.ndarray]) -> str:
    best, best_sim = None, -1.0
    if _embed_model is not None and embed_vec is not None:
        cand_vecs = _embed_model.encode(items)
        cand_vecs = np.asarray([v/(np.linalg.norm(v)+1e-12) for v in cand_vecs])
        sim = (cand_vecs @ embed_vec).reshape(-1)
        idx = int(np.argmax(sim))
        best, best_sim = items[idx], float(sim[idx])
    best_ed, best_ed_line = 10**9, None
    for s in items:
        d = _levenshtein(line, s)
        if d < best_ed:
            best_ed, best_ed_line = d, s
    return best if best is not None else best_ed_line

# ===== Embedding classifier (멀티 앵커 평균 + 인사 가드레일) =====
def classify_top_emotion(text: str) -> Tuple[str, float]:
    if _embed_model is None or not text:
        t = text.lower()
        scores = {
            "hope":     int(("희망" in t) or ("괜찮" in t) or ("고마" in t) or ("응원" in t) or bool(_GREETING_RE.search(text))),
            "trust":    int(("믿" in t) or ("안심" in t) or ("괜찮을" in t)),
            "sadness":  int(("슬픔" in t) or ("울" in t) or ("아파" in t) or ("힘들" in t)),
            "solitude": int(("외롭" in t) or ("혼자" in t) or ("고독" in t) or ("허전" in t)),
            "anger":    int(("화" in t) or ("짜증" in t) or ("미워" in t) or ("싫" in t)),
        }
        key = max(scores, key=scores.get)
        return key, float(scores[key])

    # 인사/환영류는 hope로 스냅
    if _GREETING_RE.search(text):
        return "hope", 0.99

    u = _embed_model.encode([text])[0]
    A_keys = list(EMO_ANCHOR_LISTS.keys())
    # 멀티 앵커 평균
    A_vecs = []
    for k in A_keys:
        vecs = _embed_model.encode(EMO_ANCHOR_LISTS[k])
        vecs = np.asarray(vecs)
        A_vecs.append(vecs.mean(axis=0))
    A_vecs = np.asarray(A_vecs)

    u = _norm(np.asarray(u))
    A = _norm(np.asarray(A_vecs))
    sims = (A @ u).reshape(-1)
    idx = int(np.argmax(sims))
    st.session_state.last_sims = {k: float(s) for k, s in zip(A_keys, sims)}  # 디버그
    return A_keys[idx], float(sims[idx])

# ===== Simple counter engine =====
def update_emotion_count(ss, key: str):
    """선택된 감정 하나만 +1 누적 (0에서 시작, 최대 7)."""
    ss.emotions_total[key] = float(ss.emotions_total.get(key, 0.0)) + 1.0
    ss.emo_hist.append(key)

# ===== Assets & UI =====
ASSETS_DIR = Path(__file__).parent / "assets"
MAIN_IMG = ASSETS_DIR / "main_scene.png"

st.set_page_config(page_title="7 about ...", page_icon="🧸", layout="centered")

# 공통 스타일
st.markdown("""
<style>
.block-container { max-width: 980px; margin: 0 auto; }
div.stTextArea textarea { height: 48px !important; resize: none; overflow: hidden; }
h1.app-title { text-align: center; font-size: clamp(56px, 8vw, 120px) !important;
  font-weight: 800; line-height: 1.1; margin-top: 40px; margin-bottom: 18px; }
div.stButton > button{ font-size: 30px; font-weight: 700; border: 3px solid #DADDE1;
  box-sizing: border-box; width: 260px; height: 48px; padding: 0 22px;
  border-radius: 14px; display: block; margin: 12px auto; }
@keyframes fadeUp { 0% { opacity: 0; transform: translateY(6px);} 100% { opacity: 1; transform: translateY(0);} }
.prologue-line{ opacity: 0; text-align: center; font-size: 22px; line-height: 1.7;
  margin: 6px 0; animation: fadeUp .8s ease forwards; }
.prologue-wrap{ margin-top: 48px; margin-bottom: 24px; }
.prologue-cta { opacity: 0; animation: fadeUp .8s ease forwards; }
.scene-wrap { display:flex; flex-direction:column; align-items:center; gap:18px; }
.bubble {
  width: 100% !important;         /* ← 폼과 동일 폭 */
  max-width: none !important;      /* ← 상한 해제 */
  border: 2px solid #DADDE1;
  border-radius: 18px;
  padding: 14px 18px;
  background: #fff;
  box-sizing: border-box;
}
.bubble-eebi{ border-color:#cfd6dd; }
.bubble-narr{ border-style:dashed; color:#6b7280; }
.bubble .label{ font-weight:700; color:#374151; margin-right:8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 버튼 전폭/중앙 강제 해제: 가로 정렬(컬럼 배치) 위해 폭 자동 */
div.stButton > button{
  font-size: 30px;
  font-weight: 700;
  border: 3px solid #DADDE1;
  box-sizing: border-box;
  width: auto !important;          /* ← 전폭 해제 */
  height: 48px;
  padding: 0 22px;
  border-radius: 14px;
  display: inline-block;            /* ← 가로로 자연스럽게 */
  margin: 12px 0 !important;        /* 좌우 마진은 컬럼이 담당 */
}

/* (선택) 너무 과한 전역 중앙 정렬은 유지해도 되지만,
   버튼 배치는 컬럼으로 제어하므로 이 정도만 두면 충분합니다. */
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* === (추가) 전역 중앙 정렬 === */
.block-container { text-align: center; }  /* markdown 기본 텍스트 */
div[data-testid="stMarkdown"] { text-align: center; } /* st.write/markdown 출력 */
div.stTextArea textarea { text-align: center !important; } /* 입력창 내부 텍스트 */
.bubble { text-align: center; } /* 말풍선 내부 */
.bubble .label { display: block; margin-bottom: 6px; } /* 라벨을 한 줄 위로 */
.prologue-line { text-align: center; } /* 프롤로그 문구 (이미 중앙이지만 안전차원 재명시) */

/* metric(엔딩 페이지) 숫자와 라벨 중앙 정렬 */
[data-testid="stMetric"] div { justify-content: center !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { text-align: center !important; }

/* 목록/문단/헤더도 중앙(마크다운 전역) */
div[data-testid="stMarkdown"] h1, 
div[data-testid="stMarkdown"] h2, 
div[data-testid="stMarkdown"] h3,
div[data-testid="stMarkdown"] p, 
div[data-testid="stMarkdown"] li { 
  text-align: center; 
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* === (추가) 전역 중앙 정렬 === */
.block-container { text-align: center; }  /* markdown 기본 텍스트 */
div[data-testid="stMarkdown"] { text-align: center; } /* st.write/markdown 출력 */
div.stTextArea textarea { text-align: center !important; } /* 입력창 내부 텍스트 */
.bubble { text-align: center; } /* 말풍선 내부 */
.bubble .label { display: block; margin-bottom: 6px; } /* 라벨을 한 줄 위로 */
.prologue-line { text-align: center; } /* 프롤로그 문구 (이미 중앙이지만 안전차원 재명시) */

/* metric(엔딩 페이지) 숫자와 라벨 중앙 정렬 */
[data-testid="stMetric"] div { justify-content: center !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { text-align: center !important; }

/* 목록/문단/헤더도 중앙(마크다운 전역) */
div[data-testid="stMarkdown"] h1, 
div[data-testid="stMarkdown"] h2, 
div[data-testid="stMarkdown"] h3,
div[data-testid="stMarkdown"] p, 
div[data-testid="stMarkdown"] li { 
  text-align: center; 
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 요약자 말풍선의 라벨(📜 요약자)만 숨김 */
.bubble-narr .label{ display:none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 1) 이미지 컨테이너를 50%로 줄이고 중앙 정렬 */
.scene-wrap [data-testid="stImage"]{
  width: 50% !important;
  max-width: min(320px, 45vw) !important;  /* 절반 상한 */
  margin: 0 auto !important;               /* 중앙 정렬 */
}

/* 2) 컨테이너 안의 img는 컨테이너 너비에 맞춤 */
.scene-wrap [data-testid="stImage"] img{
  width: 100% !important;
  height: auto !important;
  display: block !important;
  border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ===== State =====
def ensure_main_state():
    ss = st.session_state
    if "page" not in ss: ss.page = "title"
    if "turn" not in ss: ss.turn = 1
    if "eebi_text" not in ss: ss.eebi_text = "…안녕? 난 이비야."
    if "narr_text" not in ss: ss.narr_text = ""
    if "silent_turns" not in ss: ss.silent_turns = 0
    # 파라미터(감정) 누적치: 0에서 시작
    if "emotions_total" not in ss: ss.emotions_total = {k: 0.0 for k in EMO_KEYS}
    if "user_hist" not in ss: ss.user_hist = []
    if "eebi_hist" not in ss: ss.eebi_hist = []
    if "emo_hist" not in ss: ss.emo_hist = []           # 각 턴에서 선택된 감정 기록
    if "wl_idx" not in ss: ss.wl_idx = {k: 0 for k in EMO_KEYS}

def title_page():
    st.markdown("<h1 class='app-title'>7 about ...</h1>", unsafe_allow_html=True)
    c_sp, c1, c2 = st.columns([6, 1, 1])
    with c1:
        start = st.button("시작", key="btn_start", use_container_width=True)
    with c2:
        endings = st.button("엔딩", key="btn_endings", use_container_width=True)
    st.markdown("---")
    # _embed_model이 없어도 EMBED_MODEL_NAME은 항상 정의
    st.caption(f"Embedding: {EMBED_MODEL_NAME if _embed_model else 'keyword-backup'} • Whitelist replies • Counter endings")
    if start:
        st.session_state.page = "prologue"; st.rerun()

def prologue_page():
    lines = [
        "당신은 숲 속에 홀로 있는 곰을 발견했습니다.",
        "곰에게 말을 걸어 이야기를 해보세요",
        "대화는 총 7번 나눌 수 있습니다.",
        "이야기에 따라 곰의 감정이 변화합니다.",
        "7턴 후 최종 감정 상태에 따라 엔딩이 변화합니다.",
    ]
    st.write("")
    st.markdown("<div class='prologue-wrap'>", unsafe_allow_html=True)
    base_delay, step = 0.2, 1.0
    for i, t in enumerate(lines):
        delay = base_delay + i*step
        st.markdown(f"<p class='prologue-line' style='animation-delay:{delay:.2f}s'>{t}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    final_delay = base_delay + len(lines)*step + 0.2
    st.markdown(
        f"<div class='prologue-cta' style='animation-delay:{final_delay:.2f}s'>",
        unsafe_allow_html=True,
    )

    # ▶ 오른쪽 정렬: spacer + 버튼 컬럼
    c_sp, c_btn = st.columns([6, 1])
    with c_btn:
        ok = st.button("알겠어요!", key="btn_ok_prologue", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if ok:
        st.session_state.page = "main"; st.rerun()

def pick_whitelist_line(ss, emo_key: str) -> str:
    items = EEBI_WHITELIST.get(emo_key, ["…"])
    n = len(items)
    if n == 0: return "…"

    # 1) 하이브리드-스냅
    if USE_EEVE_SIMILAR and _http is not None and _embed_model is not None:
        try:
            cand_list = _eeve_suggest_similar(emo_key, items)
            if cand_list:
                TAU_COS, MAX_EDIT = 0.92, 6
                kept = []
                for c in cand_list:
                    c = normalize_text(c)
                    if not validate_eebi_text(c): continue
                    c_vec = _embed_model.encode([c])[0]
                    c_vec = c_vec / (np.linalg.norm(c_vec)+1e-12)
                    wl_vecs = _embed_model.encode(items)
                    wl_vecs = np.asarray([v/(np.linalg.norm(v)+1e-12) for v in wl_vecs])
                    cos_max = float((wl_vecs @ c_vec).max())
                    ed_min = min(_levenshtein(c, s) for s in items)
                    if (cos_max >= TAU_COS) or (ed_min <= MAX_EDIT):
                        kept.append((c, c_vec))
                if kept:
                    c, c_vec = kept[0]
                    snapped = _closest_whitelist(c, items, c_vec)
                    snapped = normalize_text(snapped)
                    return snapped if validate_eebi_text(snapped) else normalize_text(items[0])
        except Exception:
            pass

    # 2) LLM-Selector (인덱스만)
    if USE_EEVE_SELECTOR and _http is not None:
        try:
            k = _eeve_choose_index(n, emo_key)
            line = normalize_text(items[k])
            if validate_eebi_text(line): return line
        except Exception:
            pass

    # 3) 라운드로빈
    k = ss.wl_idx.get(emo_key, 0) % n
    ss.wl_idx[emo_key] = k + 1
    line = normalize_text(items[k])
    if not validate_eebi_text(line):
        for cand in items:
            c = normalize_text(cand)
            if validate_eebi_text(c): return c
        return "…"
    return line

def main_page():
    ensure_main_state()
    ss = st.session_state
    st.markdown("<div class='scene-wrap'>", unsafe_allow_html=True)

    # (A) 씬 이미지 — 최상단
    if MAIN_IMG.exists():
        st.image(str(MAIN_IMG))  # use_container_width / width 파라미터 생략
    else:
        st.info("씬 이미지를 찾지 못했어요. assets/main_scene.png 파일을 넣어주세요.")
        st.markdown(
            "<div style='width:min(720px,92vw);height:420px;border:2px dashed #DADDE1;"
            "border-radius:18px;display:flex;align-items:center;justify-content:center;color:#94a3b8;'>"
            "[ scene placeholder ]</div>",
            unsafe_allow_html=True
        )

    # (B) 해설자 말풍선 — 내용이 있을 때만, 이비 바로 위에 표시
    if ss.narr_text:
        st.markdown(
            f"<div class='bubble bubble-narr'><span class='label'>📜 요약자</span>{ss.narr_text}</div>",
            unsafe_allow_html=True
        )

    # (C) 이비 말풍선 — 사용자 입력 바로 위
    st.markdown(
        f"<div class='bubble bubble-eebi'><span class='label'>🐻 이비</span>{ss.eebi_text}</div>",
        unsafe_allow_html=True
    )

    with st.form("user_say", clear_on_submit=True):
        user_text = st.text_area(
            label="",
            key=f"user_text_{ss.turn}",
            max_chars=30, height=48,
            placeholder="당신의 말을 30자 이내로 적어주세요",
            label_visibility="collapsed"
        )
        # ⬇️ 컬럼/래퍼 없이 기본 형태로
        # 입력창은 그대로 두고, 제출 버튼만 우측 정렬
        col_sp, col_btn = st.columns([6, 1])
        with col_btn:
            submitted = st.form_submit_button("말하기", use_container_width=True)

    if submitted:
        txt = (user_text or "").strip()
        if txt == "" or txt in ["…", ".", "..."]:
            ss.silent_turns += 1
            ss.narr_text = "당신은 이비에게 아무 말도 건네지 않았습니다."
            ss.eebi_text = "…"
        elif violates_policy(txt):
            ss.silent_turns += 1
            ss.narr_text = "입력이 정책에 맞지 않습니다."
            ss.eebi_text = "…"
        else:
            try:
                # 1) 감정 분류 (최상위 1개)
                emo_key, sim = classify_top_emotion(txt)
                if emo_key not in EMO_KEYS:
                    emo_key, sim = "solitude", 0.5

                # 2) 카운터 방식: 선택 감정 +1
                update_emotion_count(ss, emo_key)

                # 3) 이비 대사 — 화이트리스트 기반
                first = pick_whitelist_line(ss, emo_key)
                if len(first) > 30:
                    first = split_utterance_ko(first, 30)[0]
                write_reply(first)
                ss.eebi_text = first

                # 4) 히스토리/요약
                ss.user_hist.append(txt)
                ss.eebi_hist.append(first)
                label = EMO_LABEL_KO.get(emo_key, "어떤 감정도")
                ss.narr_text = f"당신은 {label} 이야기를 했습니다."
            except Exception as e:
                ss.narr_text = "이비는 이해하지 못했습니다."
                ss.eebi_text = "…"
                st.error(f"임베딩/분류 에러: {e}")

        ss.turn += 1
        if ss.turn > 7:
            ss.page = "result"
        st.rerun()

    st.markdown(f"<div style='color:#6b7280'>현재 턴: <b>{ss.turn}</b> / 7</div>", unsafe_allow_html=True)
    if "last_sims" in ss:
        st.caption(f"디버그 - 유사도: {ss.last_sims}")
    st.markdown("</div>", unsafe_allow_html=True)

def _pick_final_ending(ss) -> str:
    """7턴 종료 후 최종 엔딩 감정 결정 (동점 시 '가장 최근에 선택된 감정' 우선)."""
    totals = ss.emotions_total
    max_v = max(totals.values()) if totals else 0.0
    cands = [k for k, v in totals.items() if v == max_v]
    # 최근 등장 감정 우선
    for ek in reversed(ss.emo_hist):
        if ek in cands:
            return ek
    # 그래도 없으면 고정 우선순위
    for ek in EMO_KEYS:
        if ek in cands:
            return ek
    return "solitude"

def result_page():
    ss = st.session_state
    st.success("대화가 끝났어요. 이비의 최종 감정 상태입니다.")
    cols = st.columns(5)
    for i, k in enumerate(EMO_KEYS):
        cols[i].metric(k, f"{int(ss.emotions_total[k])} / 7")

    # 최종 엔딩 결정
    final_key = _pick_final_ending(ss)
    title, desc = EMO_ENDINGS.get(final_key, ("엔딩: 미정", "아직 정해지지 않았습니다."))
    st.write("---")
    st.subheader(title)
    st.write(desc)

    st.write("---")
    st.write("대화 요약:")
    for i, (u,e) in enumerate(zip(ss.user_hist, ss.eebi_hist), start=1):
        st.write(f"{i}. 당신: {u}")
        st.write(f"   이비: {e}")

    c_sp, c_btn = st.columns([6, 1])
    with c_btn:
        back = st.button("처음으로", use_container_width=True)
    
    if back:
        keep = []
        for k in list(st.session_state.keys()):
            if k not in keep:
                del st.session_state[k]
        st.session_state.page = "title"
        st.rerun()

# ===== Router =====
def ensure_main_state_wrapper():
    ensure_main_state()
    # if _embed_model is None:
    #     st.warning("임베딩 모델이 없어 키워드 분류기로 동작합니다. 정확도를 높이려면 'pip install sentence-transformers' 후 재실행하세요.", icon="⚠️")
    # if USE_EEVE_SIMILAR:
    #     st.info("하이브리드-스냅 활성화: 유사 후보 제안 후 화이트리스트로 스냅합니다.", icon="🧲")
    # if USE_EEVE_SELECTOR:
    #     st.info("LLM-Selector 활성화: 화이트리스트 인덱스만 선택합니다.", icon="🧩")
    # if not USE_EEVE_SELECTOR and not USE_EEVE_SIMILAR:
    #     st.caption("LLM 옵션 비활성화: 화이트리스트 라운드로빈 모드.")

ensure_main_state_wrapper()
if st.session_state.page == "title":
    title_page()
elif st.session_state.page == "prologue":
    prologue_page()
elif st.session_state.page == "result":
    result_page()
else:
    main_page()
