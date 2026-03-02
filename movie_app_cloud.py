# -*- coding: utf-8 -*-
"""
movie_app.py - 电影雷达 3.0 (学术级精准引用 + 华语专线 + 智能推荐版)
基于 TMDB API + Groq API (qwen/qwen3-32b) 的全球新片速递与电影深度百科。
【云端适配版 for Streamlit Cloud】
- LLM：Groq API，通过 ChatOpenAI + base_url 兼容层调用
- 所有 API 密钥从 .streamlit/secrets.toml 读取
"""

import os
import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Iterator

import time
import warnings
import requests
import streamlit as st
from tmdbv3api import TMDb, Movie, Search, Genre, Person, Discover

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
        warnings.filterwarnings("ignore", message=".*renamed to.*ddgs.*", category=RuntimeWarning)
    except ImportError:
        DDGS = None

# ============== 配置 ==============
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w200"
GROQ_MODEL = "qwen/qwen3-32b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ============== API 密钥（从 secrets.toml 读取）==============
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    TMDB_API_KEY = ""

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = ""

# ============== LLM 兼容层（ChatOpenAI → Groq）==============
# 设置三个环境变量，覆盖新旧版 openai SDK 的不同读取路径
os.environ["OPENAI_API_KEY"]  = GROQ_API_KEY or "placeholder"
os.environ["OPENAI_API_BASE"] = GROQ_BASE_URL   # openai SDK v0.x
os.environ["OPENAI_BASE_URL"] = GROQ_BASE_URL   # openai SDK v1.x

@st.cache_resource
def _get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=GROQ_MODEL,
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY or "placeholder",
        temperature=0.75,
        max_tokens=8000,
    )

# ============== TMDB 初始化 ==============
try:
    _tmdb = TMDb()
    _tmdb.api_key = TMDB_API_KEY
    _tmdb.language = "zh-CN"
    _movie = Movie()
    _search = Search()
    _genre = Genre()
    _person = Person()
    _discover = Discover()
except Exception as e:
    _tmdb = _movie = _search = _genre = _person = _discover = None
    _tmdb_init_error = str(e)

# ============== 页面配置与样式 ==============
st.set_page_config(page_title="电影雷达 3.0 | TMDB", layout="wide", initial_sidebar_state="expanded")

STYLE = """
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .section-header { font-size: 1.2rem; font-weight: 600; color: #2c3e50; margin: 1rem 0 0.5rem; }
    .card-box { padding: 1rem; border-radius: 8px; background: #f8f9fa; border-left: 4px solid #1f77b4; margin: 0.5rem 0; }
    .polish-box { background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .poster-cell { vertical-align: top; }
    footer { color: #6c757d; font-size: 0.85rem; margin-top: 2rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ============== 工具函数 ==============
def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None: return default
    if isinstance(obj, dict): return obj.get(key, default)
    try: return getattr(obj, key, default)
    except Exception: return default

def _safe_list(val: Any, max_len: int = 20) -> list:
    if val is None: return []
    try:
        if isinstance(val, list): return list(val)[:max_len]
        if hasattr(val, "__iter__") and not isinstance(val, str): return list(val)[:max_len]
    except Exception: pass
    return []

def remove_think_tags(text: str) -> str:
    if not text: return ""
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    out = re.sub(r"<think>.*$", "", out, flags=re.DOTALL)
    out = re.sub(r"^.*?</think>", "", out, flags=re.DOTALL)
    return out.strip()

def groq_generate(prompt: str, system: str | None = None) -> str:
    """调用 Groq API（非流式），返回完整文本。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    if not GROQ_API_KEY:
        return "[Groq API Key 未配置，请检查 secrets.toml]"
    try:
        llm = _get_llm()
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        result = llm.invoke(messages)
        return (result.content or "").strip()
    except Exception as e:
        return f"[Groq 调用失败: {e}]"

def groq_generate_stream(prompt: str, system: str | None = None) -> Iterator[str]:
    """调用 Groq API（流式），逐块 yield 文本。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    if not GROQ_API_KEY:
        yield "[Groq API Key 未配置，请检查 secrets.toml]"
        return
    try:
        llm = _get_llm()
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        for chunk in llm.stream(messages):
            part = chunk.content or ""
            if part:
                yield part
    except Exception as e:
        yield f"[Groq 调用失败: {e}]"


# ============== 功能 1：全球新片速递 ==============
def _radar_release_in_range(release_date_str: str, window_days: int) -> bool:
    if not release_date_str or not str(release_date_str).strip(): return False
    s = str(release_date_str).strip()
    try:
        if len(s) >= 10: rd = datetime.strptime(s[:10], "%Y-%m-%d").date()
        elif len(s) == 7: rd = datetime.strptime(s + "-01", "%Y-%m-%d").date()
        elif len(s) == 4: rd = date(int(s), 1, 1)
        else: return False
    except (ValueError, TypeError): return False
    today = date.today()
    return today - timedelta(days=window_days) <= rd <= today + timedelta(days=window_days)

@st.cache_data(ttl=1800)
def fetch_tmdb_new_movies(window_days: int, mode: str) -> tuple[list[dict], str | None]:
    if _movie is None: return [], _tmdb_init_error
    out: list[dict] = []
    err = None
    today = date.today()
    gte_date = (today - timedelta(days=window_days)).strftime("%Y-%m-%d")
    lte_date = (today + timedelta(days=window_days)).strftime("%Y-%m-%d")

    if mode == "global":
        try:
            np_list = _safe_list(_safe_get(_movie.now_playing(), "results") or _movie.now_playing(), 100)
            out.extend([_movie_to_row(m) for m in np_list])
        except Exception as e: err = f"now_playing: {e}"
        try:
            up_list = _safe_list(_safe_get(_movie.upcoming(), "results") or _movie.upcoming(), 100)
            out.extend([_movie_to_row(m) for m in up_list])
        except Exception as e: err = f"{err} | upcoming: {e}" if err else f"upcoming: {e}"
    elif mode == "chinese" and _discover is not None:
        try:
            zh_raw = _discover.discover_movies({
                "with_original_language": "zh",
                "primary_release_date.gte": gte_date,
                "primary_release_date.lte": lte_date,
                "sort_by": "popularity.desc",
                "without_genres": "10770",
                "with_runtime.gte": "60",
            })
            zh_list = _safe_list(_safe_get(zh_raw, "results") or zh_raw, 100)
            out.extend([_movie_to_row(m) for m in zh_list])
        except Exception as e: err = f"discover_zh: {e}"
    else:
        return [], err

    seen_ids: set[Any] = set()
    return [x for x in out if x["id"] not in seen_ids and not seen_ids.add(x["id"])], err

@st.cache_data(ttl=86400)
def _genre_id_to_names() -> dict[int, str]:
    if _genre is None: return {}
    try:
        raw = _genre.movie_list()
        items = raw.get("genres", []) if isinstance(raw, dict) else (raw if hasattr(raw, "__iter__") else [])
        return {int(_safe_get(g, "id")): str(_safe_get(g, "name") or "") for g in items if _safe_get(g, "id") is not None}
    except Exception: return {}

def _movie_to_row(m: Any) -> dict:
    title = _safe_get(m, "title") or _safe_get(m, "name") or "未知"
    vote = _safe_get(m, "vote_average")
    score_100 = round(float(vote) * 10, 1) if vote is not None else None
    release = _safe_get(m, "release_date") or ""
    
    genres = _safe_list(_safe_get(m, "genres"), 3)
    if genres:
        names = [_safe_get(g, "name") or str(g) for g in genres]
        genre_str = "、".join(n for n in names if n) or "—"
    else:
        gids = _safe_list(_safe_get(m, "genre_ids"), 3)
        gmap = _genre_id_to_names()
        genre_str = "、".join(gmap.get(int(gid), str(gid)) for gid in gids) if gids else "—"
        
    poster_path = _safe_get(m, "poster_path")
    return {
        "id": _safe_get(m, "id"),
        "标题": title,
        "TMDB评分(百分制)": f"{score_100:.1f}" if score_100 is not None else "暂无",
        "上映日期": release,
        "分类": genre_str,
        "poster_url": (TMDB_IMAGE_BASE + poster_path) if poster_path else "",
    }

def _enrich_radar_rows(rows: list[dict], window_days: int, lang_filter: str) -> list[dict]:
    enriched = []
    for i, r in enumerate(rows):
        mid = r.get("id")
        if mid is None:
            enriched.append({**r, "导演": "—", "主演(前3)": "—", "时长(分钟)": "—", "制作国家": "—", "标题(中英)": r.get("标题", "—")})
            continue
        with st.spinner(f"正在拉取详情 ({i+1}/{len(rows)})…"):
            data = fetch_full_movie_data(mid)
            credits = fetch_movie_credits(mid)
        orig_lang = (data.get("original_language") or "").strip() if data else ""
        if lang_filter == "global" and orig_lang == "zh":
            continue
        if lang_filter == "chinese" and orig_lang != "zh":
            continue
        title_cn = data.get("title") or r.get("标题", "—") if data else r.get("标题", "—")
        orig_title = (data.get("original_title") or "").strip() if data else ""
        if orig_lang and orig_title and orig_lang != "zh":
            title_display = f"{title_cn} ({orig_title})"
        else:
            title_display = title_cn
        country = "、".join(data.get("origin_country", [])[:3]) if data and data.get("origin_country") else "—"
        # 过滤非电影内容（仅华语模式）：含晚会/演出关键词或时长 < 60 分钟
        if lang_filter == "chinese":
            _NON_MOVIE_KW = ['联欢晚会', '春晚', '演唱会', '音乐会', 'concert', 'Concert', '晚会', 'gala', 'Gala', '颁奖典礼', '庆典晚会', '相声专场']
            _t = (title_cn or "") + (orig_title or "")
            if any(kw in _t for kw in _NON_MOVIE_KW):
                continue
            _rt = data.get("runtime") if data else None
            if _rt is not None and int(_rt) < 60:
                continue
        enriched.append({
            **r,
            "标题(中英)": title_display,
            "导演": "、".join(credits.get("directors", [])[:3]) if credits.get("directors") else "—",
            "主演(前3)": "、".join(credits.get("cast_top5", [])[:3]) if credits.get("cast_top5") else "—",
            "时长(分钟)": str(data.get("runtime", "—")) if data and data.get("runtime") else "—",
            "制作国家": country,
        })
    def _sort_key(x: dict) -> float:
        s = x.get("TMDB评分(百分制)")
        return float(s) if s != "暂无" else -1.0
    enriched.sort(key=_sort_key, reverse=True)
    return enriched

def _render_radar_grid(rows: list[dict], window_days: int):
    st.markdown(f"<p class='section-header'>📂 新片海报墙（±{window_days} 天内上映）</p>", unsafe_allow_html=True)
    for i in range(0, len(rows), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j >= len(rows): break
            r = rows[i + j]
            with cols[j]:
                if r.get("poster_url"): st.image(r["poster_url"], width=140)
                else: st.markdown("*(暂无海报)*")
                st.caption(f"**{r.get('标题(中英)', r.get('标题', '—'))}**")
                st.caption(f"TMDB {r.get('TMDB评分(百分制)', '—')} · {r.get('上映日期', '—')}")
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(
        [{"标题": r.get("标题(中英)"), "评分": r.get("TMDB评分(百分制)"), "上映": r.get("上映日期"), "分类": r.get("分类"), "导演": r.get("导演"), "主演(前3)": r.get("主演(前3)"), "时长(分钟)": r.get("时长(分钟)"), "国家": r.get("制作国家")} for r in rows],
        width="stretch", hide_index=True,
    )

def run_global_radar(window_days: int):
    st.markdown("<p class='main-header'>🎬 全球新片速递 (Global Radar) · 电影雷达 3.0</p>", unsafe_allow_html=True)
    if st.button("🔄 获取新片列表", key="radar_global_refresh"):
        with st.spinner("正在从 TMDB 获取全球新片（非华语）…"):
            rows, api_err = fetch_tmdb_new_movies(window_days, "global")
        if api_err: st.warning(f"部分接口异常: {api_err}")
        if not rows:
            st.warning("未获取到电影列表，请检查 TMDB 配置与网络。")
            return
        rows = [r for r in rows if _radar_release_in_range(r.get("上映日期") or "", window_days)]
        if not rows:
            st.warning(f"当前 ±{window_days} 天内无符合上映日期的电影，请尝试在左侧增加时间范围。")
            return
        rows = _enrich_radar_rows(rows, window_days, "global")
        if not rows:
            st.warning("当前时间范围内无符合条件的非华语新片。")
            return
        _render_radar_grid(rows, window_days)
    else:
        st.info(f"点击上方按钮从 TMDB 获取全球新片（仅展示原语言非中文、±{window_days} 天内上映影片）。")

def run_chinese_radar(window_days: int):
    st.markdown("<p class='main-header'>🎬 华语新片速递 (Chinese Radar) · 电影雷达 3.0</p>", unsafe_allow_html=True)
    if st.button("🔄 获取新片列表", key="radar_chinese_refresh"):
        with st.spinner("正在从 TMDB 获取华语新片…"):
            rows, api_err = fetch_tmdb_new_movies(window_days, "chinese")
        if api_err: st.warning(f"部分接口异常: {api_err}")
        if not rows:
            st.warning("未获取到华语电影列表，请检查 TMDB 配置与网络。")
            return
        rows = [r for r in rows if _radar_release_in_range(r.get("上映日期") or "", window_days)]
        if not rows:
            st.warning(f"当前 ±{window_days} 天内无符合上映日期的华语电影，请尝试在左侧增加时间范围。")
            return
        rows = _enrich_radar_rows(rows, window_days, "chinese")
        if not rows:
            st.warning("当前时间范围内无符合条件的华语新片。")
            return
        _render_radar_grid(rows, window_days)
    else:
        st.info(f"点击上方按钮从 TMDB 获取华语新片（仅展示原语言为中文、±{window_days} 天内上映影片）。")

# ============== 功能 2：电影深度百科 ==============
def _get_search_candidates(name: str, max_results: int = 5) -> list[dict]:
    if not _search or not name.strip(): return []
    try:
        raw = _search.movies(name.strip())
        return [{"id": int(_safe_get(m, "id")), 
                 "title": _safe_get(m, "title") or _safe_get(m, "name") or "", 
                 "year": (_safe_get(m, "release_date") or "")[:4], 
                 "poster_url": (TMDB_IMAGE_BASE + _safe_get(m, "poster_path")) if _safe_get(m, "poster_path") else ""}
                for m in _safe_list(_safe_get(raw, "results") or raw, max_results) if _safe_get(m, "id")]
    except Exception: return []

def fetch_full_movie_data(movie_id: int) -> dict | None:
    if _movie is None: return None
    try: details = _movie.details(movie_id)
    except Exception: return None
    prod_names = [str(_safe_get(p, "name")).strip() for p in _safe_list(_safe_get(details, "production_companies"), 5) if _safe_get(p, "name")][:5]
    countries = [str(_safe_get(c, "name")).strip() for c in _safe_list(_safe_get(details, "production_countries"), 10) if _safe_get(c, "name")]
    return {
        "title": _safe_get(details, "title") or _safe_get(details, "name") or "",
        "original_title": _safe_get(details, "original_title") or "",
        "original_language": _safe_get(details, "original_language") or "",
        "overview": _safe_get(details, "overview") or "",
        "runtime": int(_safe_get(details, "runtime") or 0),
        "release_date": _safe_get(details, "release_date") or "",
        "production_companies": prod_names,
        "origin_country": countries or ["—"],
    }

def fetch_movie_credits(movie_id: int) -> dict:
    out: dict = {"directors": [], "director_ids": [], "producers": [], "cast_top5": []}
    if _movie is None: return out
    try: credits = _movie.credits(movie_id)
    except Exception: return out
    
    crew_list = credits.get("crew", []) if isinstance(credits, dict) else _safe_list(_safe_get(credits, "crew"), 50)
    for c in crew_list:
        job = (_safe_get(c, "job") or "").strip()
        name = str(_safe_get(c, "name") or "").strip()
        if job == "Director":
            out["directors"].append(name)
            if _safe_get(c, "id") is not None: out["director_ids"].append(int(_safe_get(c, "id")))
        elif job == "Producer":
            out["producers"].append(name)
            
    cast_list = credits.get("cast", []) if isinstance(credits, dict) else _safe_list(_safe_get(credits, "cast"), 5)
    out["cast_top5"] = [str(_safe_get(c, "name") or "").strip() for c in cast_list]
    out["directors"] = [n for n in out["directors"] if n]
    out["producers"] = [n for n in out["producers"] if n][:5]
    return out

def fetch_director_top_movies(director_id: int, top_n: int = 5, exclude_movie_id: int | None = None) -> list[dict]:
    """同时获取同导演作品的 official overview 以供 AI 写推荐语"""
    if _person is None or not director_id: return []
    try:
        raw = _person.movie_credits(director_id)
        seen, out = set(), []
        for key in ("crew", "cast"):
            for m in _safe_list(_safe_get(raw, key), 80):
                mid = _safe_get(m, "id")
                if not mid or mid in seen or (exclude_movie_id and int(mid) == exclude_movie_id): continue
                seen.add(int(mid))
                title = _safe_get(m, "title") or _safe_get(m, "name") or ""
                overview = _safe_get(m, "overview") or "暂无简介"
                poster_path = _safe_get(m, "poster_path")
                if title: out.append({"id": int(mid), "title": title, "popularity": float(_safe_get(m, "popularity") or 0.0), "overview": overview, "poster_url": (TMDB_IMAGE_BASE + poster_path) if poster_path else ""})
        out.sort(key=lambda x: x.get("popularity", 0), reverse=True)
        return out[:top_n]
    except Exception: return []

def fetch_movie_recommendations(movie_id: int, top_n: int = 5) -> list[dict]:
    """同时获取相似作品的 official overview 以供 AI 写推荐语"""
    if _movie is None: return []
    try:
        raw = _movie.recommendations(movie_id)
        out = []
        for m in _safe_list(_safe_get(raw, "results") or raw, top_n):
            mid, title = _safe_get(m, "id"), _safe_get(m, "title") or _safe_get(m, "name") or ""
            overview = _safe_get(m, "overview") or "暂无简介"
            if not mid or not title: continue
            director = ""
            try:
                creds = _movie.credits(int(mid))
                crew = creds.get("crew", []) if isinstance(creds, dict) else _safe_list(_safe_get(creds, "crew"), 30)
                director = next((str(_safe_get(c, "name") or "").strip() for c in crew if (_safe_get(c, "job") or "").strip() == "Director"), "—")
            except Exception: pass
            poster_path = _safe_get(m, "poster_path")
            out.append({"id": int(mid), "title": title, "director": director, "overview": overview, "poster_url": (TMDB_IMAGE_BASE + poster_path) if poster_path else ""})
        return out[:top_n]
    except Exception: return []

def _ddgs_client():
    try: return DDGS(impersonate=None)
    except TypeError: return DDGS()

def multi_directional_search(movie_name: str, year: str, director_name: str = "", is_foreign: bool = False) -> tuple[dict[str, list[str]], dict[str, dict]]:
    out: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": [], "E": []}
    sources_map: dict[str, dict] = {} 
    source_idx = 1
    
    if not DDGS or not movie_name or not movie_name.strip(): return out, sources_map
    name, director, y = movie_name.strip(), (director_name or "").strip(), (year or "").strip()[:4]
    base = f"{name} {director}".strip()
    
    queries = [("A", f"{base} {y} 创作背景 幕后花絮 采访"), ("B", f"{base} 深度解析 艺术风格 影评"), ("C", f"{base} {y} 分析 影评")]
    if is_foreign and base:
        queries.extend([("D", f"Analysis of {name} by {director} film"), ("E", f"Review of {name} cinematography")])

    def _is_relevant(title_text: str, body_text: str) -> bool:
        t, b = (title_text or "").lower(), (body_text or "").lower()
        if not name: return True
        name_parts = [w.strip() for w in re.split(r"[\s\-·,，、]+", name) if len(w.strip()) >= 2]
        if name in t or name in b: return True
        if any(p and (p in t or p in b) for p in name_parts): return True
        if director and (director in t or director in b): return True
        return False

    for key, q in queries:
        max_per_query = 6 if key in ("A", "B", "C") else 3
        try:
            with _ddgs_client() as ddgs:
                results = list(ddgs.text(q, max_results=max_per_query))
            for r in results[:max_per_query]:
                title, body, href = r.get("title", ""), r.get("body", ""), r.get("href") or r.get("url") or ""
                if not _is_relevant(title, body): continue
                
                if href and not any(s['url'] == href for s in sources_map.values()):
                    ref_id = f"[{source_idx}]"
                    sources_map[ref_id] = {"title": title or "(无标题)", "url": href}
                    out[key].append(f"【{ref_id}】[{title}] {body[:400]}")
                    source_idx += 1
            time.sleep(0.5)
        except Exception: pass
    return out, sources_map

def run_deep_analysis():
    if "deep_candidates" not in st.session_state: st.session_state.deep_candidates = []
    if "deep_movie_id" not in st.session_state: st.session_state.deep_movie_id = None
    if "deep_run_analysis" not in st.session_state: st.session_state.deep_run_analysis = False

    st.markdown("<p class='main-header'>📚 电影深度百科 (Deep Analysis) · 电影雷达 3.0</p>", unsafe_allow_html=True)
    movie_name = st.text_input("输入电影名称（中英文均可）", key="deep_movie_name", placeholder="例如：奥本海默")

    col_search, col_reset = st.columns([2, 1])
    with col_search:
        if st.button("🔍 搜索", key="deep_search_btn"):
            if movie_name and movie_name.strip():
                with st.spinner("正在 TMDB 搜索…"):
                    st.session_state.deep_candidates = _get_search_candidates(movie_name.strip(), 5)
                st.session_state.deep_movie_id = None
                st.session_state.deep_run_analysis = False
            else: st.warning("请输入电影名称。")
    with col_reset:
        if st.button("重新选择", key="deep_reset_btn"):
            st.session_state.deep_candidates = []
            st.session_state.deep_movie_id = None
            st.session_state.deep_run_analysis = False
            st.rerun()

    candidates, about_to_run = st.session_state.deep_candidates, st.session_state.deep_run_analysis and st.session_state.deep_movie_id is not None

    if candidates and not about_to_run:
        st.markdown("<p class='section-header'>选择要分析的电影</p>", unsafe_allow_html=True)
        labels = [f"{c['title']} ({c.get('year') or '年份未知'})" for c in candidates]
        selected_index = st.radio("候选电影", range(len(candidates)), format_func=lambda i: labels[i], key="deep_radio", horizontal=False)
        cols = st.columns(min(5, len(candidates)))
        for i, col in enumerate(cols):
            if i >= len(candidates): break
            with col:
                if candidates[i].get("poster_url"): st.image(candidates[i]["poster_url"], width=100)
                else: st.markdown("*(暂无海报)*")
                st.caption(f"**{candidates[i]['title']}**\n\n{candidates[i].get('year', '—') or '—'}")
        st.markdown("---")
        if st.button("确认分析这部", key="deep_confirm_btn"):
            st.session_state.deep_movie_id = candidates[selected_index]["id"]
            st.session_state.deep_run_analysis = True
            st.rerun()

    if about_to_run:
        movie_id = st.session_state.deep_movie_id
        st.session_state.deep_run_analysis = False

        data = fetch_full_movie_data(movie_id)
        if data is None:
            st.error("获取电影详情失败，请稍后重试。")
            return
        credits = fetch_movie_credits(movie_id)
        
        year = (data.get("release_date") or "")[:4]
        is_foreign = (data.get("original_language") or "").strip() != "zh"
        director_str = ", ".join(credits["directors"]) if credits.get("directors") else ""
        
        with st.spinner("多重搜索 + 导演代表作 + 相似推荐…"):
            snippets, sources_map = multi_directional_search(data["title"], year, director_str, is_foreign)
            st.session_state.deep_search_sources = sources_map
            director_top5 = fetch_director_top_movies(credits["director_ids"][0], 5, exclude_movie_id=movie_id) if credits.get("director_ids") else []
            recs = fetch_movie_recommendations(movie_id, 5)

        # 把这10部电影的官方简介拼装好，作为“弹药”喂给 AI
        # ── 构建主分析的数据块（不含推荐语，专注正文三板块）──────────
        tmdb_block = f"""
【TMDB 官方硬数据 - 人名/日期等必须严格以此为准，禁止编造】
片名: {data['title']} ({data.get('original_title', '') or '—'})
上映日期: {data.get('release_date', '') or '—'}
剧情大纲: {data['overview']}
国别: {"、".join(data.get("origin_country", [])) or "—"}
导演: {director_str or "—"}
主演(前5): {", ".join(credits['cast_top5']) if credits['cast_top5'] else "—"}
"""
        search_block = ""
        for key in ("A", "B", "C", "D", "E"):
            if snippets.get(key): search_block += "\n".join(snippets[key]) + "\n\n"
        if search_block:
            search_block = "\n【网页搜索片段（带有 [序号] 标记） - 请严格使用该序号进行引用】\n" + search_block

        # ── 请求一：主分析（SUMMARY + STORY_BACKGROUND + CRITIC_REVIEW）──
        system_main = (
            "你是一位兼具学术深度与文学才华的电影百科作家，文风细腻、论述严谨、见解独到。"
            "你的写作应当像顶尖电影杂志的长篇深度稿件，而非教科书式的条目罗列。"
            "【语言风格】行文要有温度和个性，善用具体细节、感性描写与理性分析的交织。"
            "严禁使用以下空洞套语：'非常出色''非常重要''非常独特''呈现出一种非常真实和自然的'"
            "'使得这部电影成为一部经典''导演的拍摄手法和演员的演技''具有非常强的文化和历史意义'。"
            "每一个判断都必须有具体事实或细节支撑，不得发表无内容的泛泛评价。"
            "【事实准确性】凡涉及人名、日期、奖项，必须严格以 TMDB 官方数据为准，禁止编造。"
            "【引用规则】若某句话的具体内容直接来自网页搜索片段，在该句句尾标注 [n]；"
            "综合分析和个人判断无需标注；严禁使用不存在于搜索素材中的编号。"
            "【语言要求】所有输出使用纯中文，专有名词首次出现时附中文说明。"
            "【格式要求】以流动的散文段落书写，段落间自然换行，不使用任何编号或项目符号。"
        )
        prompt_main = f"""请根据以下「TMDB 官方数据」与「网页搜索片段」，为电影《{data['title']}》撰写一篇极为深入的长篇电影百科。
{tmdb_block}
{search_block}
【写作要求】

[SUMMARY] 板块（约 150 字）：
用精炼而有文学感的语言勾勒这部电影的灵魂——它讲述了什么、触动了什么、在电影史上意味着什么。不要泛泛地说"艺术成就很高"，而要落到具体：某个场景、某种情感、某段历史。

[STORY_BACKGROUND] 板块（不少于 800 字）：
以叙事笔法带领读者走入这部电影诞生的时代与背景。从当时的社会土壤和历史气候写起，自然过渡到导演萌生这个创作念头的契机与心路。剧本是如何一稿稿打磨成形的？主演的选定经历了怎样的考量？拍摄现场有哪些值得记录的细节或挑战？资金筹措和发行历程如何影响了最终面貌？让整个板块读起来像一段有起伏、有温度的幕后故事，而非各维度的逐条汇报。引用搜索素材中的具体事实时，在句尾标注 [n]；综合分析无需标注，严禁编造引用编号。

[CRITIC_REVIEW] 板块（不少于 800 字）：
以资深影评人的眼光，对这部电影展开有观点、有锋芒的深度解读。不要逐项打分，而要找到这部电影最核心的艺术张力——它的摄影语言、叙事节奏、主题深度、表演层次——彼此之间如何共振，构成了怎样独特的观影体验。将它放置在同类影片乃至整个电影史的坐标中，说清楚它的独特位置和历史影响。对主演表演的分析要有具体的场景依据，不能只说"表演出色"。引用搜索素材时在句尾标注 [n]，综合判断无需标注，严禁编造引用编号。

必须且仅包含以下三个标记块（标签严格独占一行，供程序解析）：
[SUMMARY]
[STORY_BACKGROUND]
[CRITIC_REVIEW]"""

        result_holder = []
        with st.spinner("🧠 深度检索与撰写中，这可能需要一段时间，请稍候…"):
            for chunk in groq_generate_stream(prompt_main, system=system_main):
                result_holder.append(chunk)
        analysis = "".join(result_holder)

        if "[Groq 调用失败" in analysis:
            st.error(analysis)
            return

        clean = remove_think_tags(analysis)

        def extract_section(text: str, tag: str, next_tags: list) -> str:
            """更健壮的块解析：在 tag 和下一个出现的任意标签之间取内容"""
            if tag not in text:
                return ""
            after = text.split(tag, 1)[1]
            earliest = len(after)
            for nt in next_tags:
                idx = after.find(nt)
                if idx >= 0 and idx < earliest:
                    earliest = idx
            return after[:earliest].strip()

        MAIN_TAGS = ["[SUMMARY]", "[STORY_BACKGROUND]", "[CRITIC_REVIEW]"]
        summary_raw   = extract_section(clean, "[SUMMARY]",          [t for t in MAIN_TAGS if t != "[SUMMARY]"])
        story_bg      = extract_section(clean, "[STORY_BACKGROUND]", ["[CRITIC_REVIEW]"])
        critic_review = extract_section(clean, "[CRITIC_REVIEW]",    [])

        # 容错处理：关键内容缺失时回退
        if not story_bg and not critic_review:
            fallback = clean
            for tag in MAIN_TAGS:
                fallback = fallback.replace(tag, "")
            story_bg = fallback.strip()
        critic_review = critic_review or ""
        summary_raw   = summary_raw or (clean[:300] if len(clean) > 5 else data.get("overview", "")[:200])

        # ── 请求二：推荐语（director_recs + similar_recs）────────────────
        dir_recs_ctx = "\n".join([
            f"{i+1}. 《{m['title']}》\n   TMDB简介：{m.get('overview', '暂无简介')}"
            for i, m in enumerate(director_top5)
        ]) if director_top5 else "暂无"

        sim_recs_ctx = "\n".join([
            f"{i+1}. 《{m['title']}》（导演：{m.get('director', '未知')}）\n   TMDB简介：{m.get('overview', '暂无简介')}"
            for i, m in enumerate(recs)
        ]) if recs else "暂无"

        director_recs_text = ""
        similar_recs_text = ""

        if director_top5 or recs:
            system_recs = (
                "你是一位有个人风格的电影推荐人，擅长用精准而有温度的语言点亮每部电影的独特魅力。"
                "【核心原则】每部电影的推荐语必须从该电影的 TMDB 简介出发，"
                "抓住这部电影最与众不同的那个核心——可以是一个反转、一段关系、一种视觉风格、"
                "一个历史切口，或一种令人难忘的情感体验——围绕它展开，让读者感受到这部电影的唯一性。"
                "【严格禁止以下万能句式】："
                "'通过对……的描绘，影片展现了……'"
                "'带领观众踏上了一段……的旅程'"
                "'导演的拍摄手法和演员的演技使得这部电影成为一部经典'"
                "'探讨了……等主题''具有很高的历史地位和深远影响'。"
                "每句推荐语必须有只属于这部电影的具体内容，换一部电影就不成立。"
                "所有输出使用纯中文。"
            )
            prompt_recs = f"""请为以下两组电影各写推荐语，素材来自 TMDB 官方简介。

【第一组：导演 {director_str or "该导演"} 的其他代表作】
{dir_recs_ctx}

【第二组：与《{data['title']}》主题相似的推荐电影】
{sim_recs_ctx}

要求：
- 每部电影写 2-3 句推荐语，语言生动，揭示该片独特价值与观影理由
- 必须依据上方 TMDB 简介中的具体内容来写，体现每部电影的差异化特质
- 严禁使用"导演的拍摄手法和演员的演技使得这部电影成为一部经典的XXX片"等万能模板
- 两组推荐语内容完全独立，不得相互重复
- 每部电影推荐语之间空一行

必须且仅包含以下两个标记块（标签独占一行）：
[director_recs]
（按第一组列表顺序，逐部写推荐语，格式："- 《片名》：推荐语"）
[similar_recs]
（按第二组列表顺序，逐部写推荐语，格式："- 《片名》：推荐语"）"""

            with st.spinner("✍️ 生成推荐语…"):
                recs_raw = groq_generate(prompt_recs, system=system_recs)

            if "[Groq 调用失败" not in recs_raw:
                recs_clean = remove_think_tags(recs_raw)
                REC_TAGS = ["[director_recs]", "[similar_recs]"]
                director_recs_text = extract_section(recs_clean, "[director_recs]", ["[similar_recs]"])
                similar_recs_text  = extract_section(recs_clean, "[similar_recs]",  [])
                # 防重复：若两段完全相同说明解析错误
                if similar_recs_text and similar_recs_text == director_recs_text:
                    similar_recs_text = ""

        # 一句话简介直接使用 TMDB 官方简介，准确且无需额外 API 调用
        tmdb_overview = (data.get("overview") or "").strip()
        one_liner = tmdb_overview if tmdb_overview else summary_raw[:200]

        def render_movie_poster_row(movies: list, max_count: int = 5):
            """横排展示最多 max_count 张电影海报 + 片名"""
            display = [m for m in movies if m][:max_count]
            if not display:
                return
            cols = st.columns(max_count)
            for i, col in enumerate(cols):
                with col:
                    if i < len(display):
                        m = display[i]
                        poster = m.get("poster_url", "")
                        if poster:
                            st.image(poster, width=150)
                        else:
                            st.markdown("<div style='height:140px;background:#eee;border-radius:6px;"
                                        "display:flex;align-items:center;justify-content:center;"
                                        "font-size:12px;color:#999'>暂无海报</div>", unsafe_allow_html=True)
                        st.caption(f"**{m.get('title', '—')}**")
                    else:
                        st.empty()

        def _parse_recs_to_dict(text: str) -> dict:
            """
            Parse AI recommendation text into {normalized_title: body_text} dict.
            Handles formats: "1. 《Title》：...", "- 《Title》：...", "《Title》：..."
            Returns dict keyed by title string (stripped of brackets).
            """
            import re
            result: dict = {}
            raw_lines = [l.strip() for l in text.splitlines()]
            item_start = re.compile(r'^(\d+[.。]\s*|[-•]\s*)?《(.+?)》')
            current_title: str | None = None
            buf: list[str] = []
            def _flush():
                if current_title and buf:
                    body = " ".join(buf).strip()
                    # Remove leading colon/dash after title
                    body = re.sub(r'^[：:—\-\s]+', '', body)
                    # Also strip leading director info if AI included it: （导演：XXX）：
                    body = re.sub(r'^（导演：[^）]+）[：:\s]*', '', body)
                    body = re.sub(r'^导演：[^：：\n]+[：:：]\s*', '', body)
                    result[current_title] = body
            for line in raw_lines:
                m = item_start.match(line)
                if m:
                    _flush()
                    current_title = m.group(2).strip()
                    rest = line[m.end():]  # get remainder after title
                    # Remainder after 《Title》 on same line
                    rest = re.sub(r'^[）)（(（\s]*', '', rest)  # strip open paren from title end
                    rest = re.sub(r'^（导演：[^）]+）[：:\s]*', '', rest)  # strip director tag
                    rest = re.sub(r'^[）)（(（\s]*', '', rest)  # strip trailing bracket of title
                    rest = re.sub(r'^[：:—\-\s]+', '', rest)
                    buf = [rest] if rest else []
                elif current_title:
                    buf.append(line)
            _flush()
            return result

        def build_ordered_recs_html(recs_text: str, movies: list, show_director: bool = False) -> str:
            """
            Render recommendations as an ordered list aligned to the `movies` list order
            (which is the same order as the posters displayed above).
            For each movie in `movies`, find the matching rec in AI output by title.
            Falls back to TMDB overview if AI didn't cover that movie.
            """
            import re
            ai_map = _parse_recs_to_dict(recs_text)

            def _best_match(title: str) -> str | None:
                """Fuzzy-match a movie title against ai_map keys."""
                if title in ai_map:
                    return ai_map[title]
                # Try partial match: if any key contains the title or vice-versa
                for key, val in ai_map.items():
                    if title in key or key in title:
                        return val
                # Try matching on first 2+ CJK chars
                short = re.sub(r'[\s：:（()）《》]', '', title)[:4]
                if short:
                    for key, val in ai_map.items():
                        if short in re.sub(r'[\s：:（()）《》]', '', key):
                            return val
                return None

            html = "<ol style='padding-left:1.4em;line-height:1.9;'>"
            for m in movies:
                title = m.get('title', '—')
                director = m.get('director', '')
                rec_body = _best_match(title)
                if not rec_body:
                    # Fallback: use TMDB overview (first 2 complete sentences)
                    import re as _re
                    overview = m.get('overview', '暂无简介')
                    # Split on sentence-ending punctuation, keep delimiter
                    sentences = _re.split(r'(?<=[。！？.!?])\s*', overview.strip())
                    sentences = [s.strip() for s in sentences if s.strip()]
                    rec_body = ''.join(sentences[:3]) if sentences else overview
                # Build label
                if show_director and director and director != '—':
                    label = f"《{title}》（导演：{director}）"
                else:
                    label = f"《{title}》"
                html += f"<li style='margin-bottom:0.6em;'><strong>{label}</strong>：{rec_body}</li>"
            html += "</ol>"
            return html
        def strip_excess_citations(text, max_ref):
            import re
            if max_ref <= 0:
                return re.sub(r'\[\d+\]', '', text)
            return re.sub(r'\[(\d+)\]', lambda m: m.group(0) if int(m.group(1)) <= max_ref else '', text)

        def render_analysis_with_headers(story, review, raw_clean, max_ref):
            import re
            def _fmt(text):
                cleaned = strip_excess_citations(text, max_ref)
                # Strip AI numbered sub-headers like "1. **标题**" leaving content
                cleaned = re.sub(r'\d+\.\s+\*\*[^*]+\*\*\s*', '', cleaned)
                # Strip remaining bold markers **text** -> text
                cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
                return cleaned.replace(chr(10), '<br/>')
            if story and review:
                st.markdown("<p class='section-header' style='margin-top:0.8rem'>📖 创作背景</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-box'>{_fmt(story)}</div>", unsafe_allow_html=True)
                st.markdown("<p class='section-header' style='margin-top:1.2rem'>🎬 专业影评</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-box'>{_fmt(review)}</div>", unsafe_allow_html=True)
            else:
                combined = (story + "\n\n" + review).strip() or raw_clean
                st.markdown(f"<div class='card-box'>{_fmt(combined)}</div>", unsafe_allow_html=True)

        # ============== 渲染输出 ==============
        sources_map_now = st.session_state.get("deep_search_sources", {})
        max_ref_num = len(sources_map_now)

        st.markdown("---")
        st.markdown("<p class='section-header'>🎬 TMDB官方简介</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='polish-box'>{one_liner.replace(chr(10), '<br/>')}</div>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>📄 完整深度百科</p>", unsafe_allow_html=True)
        render_analysis_with_headers(story_bg, critic_review, clean, max_ref_num)




        if director_top5:
            st.markdown(f"<p class='section-header'>🎬 导演 {director_str} 的其他代表作</p>", unsafe_allow_html=True)
            render_movie_poster_row(director_top5, max_count=5)
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                build_ordered_recs_html(director_recs_text, director_top5, show_director=False),
                unsafe_allow_html=True
            )

        if recs:
            st.markdown("<p class='section-header'>📌 主题相似电影推荐</p>", unsafe_allow_html=True)
            render_movie_poster_row(recs, max_count=5)
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                build_ordered_recs_html(similar_recs_text, recs, show_director=True),
                unsafe_allow_html=True
            )





        sources_map = st.session_state.get("deep_search_sources", {})
        if sources_map:
            st.markdown("---")
            st.markdown("<p class='section-header'>🌐 参考来源</p>", unsafe_allow_html=True)
            for ref_id, s_info in sources_map.items():
                st.markdown(f"- **{ref_id}** [{s_info['title']}]({s_info['url']})")

    elif not candidates and not (movie_name and movie_name.strip()):
        st.info("请输入电影名称并点击「搜索」，从候选中选择一部后点击「确认分析这部」。")

# ============== 主入口 ==============
def main():
    st.sidebar.title("电影雷达 3.0")
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("选择功能", [
        "全球新片速递 (Global Radar)",
        "华语新片速递 (Chinese Radar)",
        "电影深度百科 (Deep Analysis)"
    ])
    st.sidebar.markdown("---")
    radar_window = st.sidebar.slider("新片统计范围 (天)", min_value=7, max_value=60, value=14, step=1, help="全球/华语新片速递均使用此时间范围")
    st.sidebar.markdown("---")
    st.sidebar.caption("数据：TMDB API · AI：Groq qwen3-32b")

    if _tmdb is None: st.sidebar.error("TMDB 初始化失败，请检查 API Key 与依赖。")

    if "全球新片速递" in mode: run_global_radar(radar_window)
    elif "华语新片速递" in mode: run_chinese_radar(radar_window)
    else: run_deep_analysis()

    st.markdown("<footer>数据来源于 TMDB，仅供参考。请以影院与平台为准。</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()