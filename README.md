# 电影雷达 · Movie Radar

一款电影信息与深度分析平台，整合 TMDB 电影数据库与 Groq 大模型，提供院线新片速递与学术级深度百科生成。由本人主导需求与架构，通过与 AI 协作完成开发，已部署至 Streamlit Cloud 与 HuggingFace Spaces，可公开访问。

A movie discovery and in-depth analysis platform integrating the TMDB database with Groq large language models, delivering real-time new-release tracking and AI-generated long-form film profiles. Designed and directed by me, implemented through LLM collaboration, and deployed to Streamlit Cloud and HuggingFace Spaces.

🌐 **[Streamlit Cloud](https://my-movie-ai.streamlit.app/)** · **[HuggingFace Spaces](https://huggingface.co/spaces/chnlqsray/movie-radar)**

▶ **[YouTube 演示视频](https://youtu.be/nY3jBSJbu5g)** · **[Bilibili 演示视频](https://www.bilibili.com/video/BV1WScyzDED9/)**

---

## 产品功能 · Features

**全球新片速递**：自动拉取 TMDB 正在上映与即将上映的影片，以海报墙与结构化表格双重形式呈现，按评分降序排列，支持按年份与类型筛选。

**Global new-release tracker**: Automatically fetches currently showing and upcoming films from TMDB, presented as both a poster wall and a structured table, sorted by rating, with year and genre filters.

**华语新片专线**：在 TMDB 数据质量层面，精准筛选原始语言为中文的院线电影，在代码层过滤演唱会、晚会、综艺等非电影内容，解决 TMDB 华语数据噪声问题。

**Chinese-language filter**: At the data-quality level, precisely filters films with Chinese as original language, then applies code-level filtering to remove concerts, variety shows, and other non-film content — addressing the noise in TMDB's Chinese-language dataset.

**电影深度百科**：以五维网络搜索（创作背景、艺术风格、中英文影评、摄影语言）驱动 AI 生成 1,600 字以上的结构化深度文章，包含引用来源标注，风格接近学术影评。

**Deep film profiles**: Five-axis web research (production background, artistic style, Chinese and English reviews, cinematography) drives AI generation of 1,600+ character structured articles with source attribution, in a style approaching academic film criticism.

**导演关联推荐**：自动推荐导演其他代表作与主题相似影片，每部配 AI 生成的个性化推荐语。

**Director-linked recommendations**: Automatically suggests the director's other works and thematically similar films, each with an AI-generated personalised recommendation note.

**引用系统**：搜索结果以结构化标签形式注入 prompt，LLM 在生成文章时同步输出引用编号，最终在页面底部渲染引用列表，解决 LLM 在长文生成中引用来源不稳定的问题。

**Citation system**: Search results are injected into the prompt as structured tags; the LLM outputs citation numbers inline during generation, which are rendered as a reference list at the bottom of the page — addressing LLM citation instability in long-form generation.

---

## 技术栈 · Tech Stack

| 类别 | 依赖 |
|------|------|
| 界面框架 | Streamlit |
| LLM 接入层 | langchain-openai (ChatOpenAI → Groq API, qwen/qwen3-32b) |
| 电影数据 | TMDB API (tmdbv3api) |
| 网络搜索 | DuckDuckGo Search (ddgs / duckduckgo_search 双兼容) |
| 数据处理 | Pandas |
| 自动化运维 | GitHub Actions, Playwright（保活脚本托管于 [finance_app_cloud](https://github.com/chnlqsray/finance_app_cloud)）|

---

## 部署配置 · Deployment

本项目部署于 Streamlit Cloud 与 HuggingFace Spaces。运行需在 `.streamlit/secrets.toml` 中配置以下密钥：

This project is deployed on Streamlit Cloud and HuggingFace Spaces. The following API keys must be configured in `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_tmdb_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

---

## 工程决策记录 · Engineering Notes

**为什么选 qwen3-32b 而非 llama？**

电影深度百科对中英文混合写作、长文结构一致性与引用格式遵循的要求较高。实测 qwen3-32b 在中文写作质量与指令遵循（尤其是引用编号格式）方面显著优于同规模 llama 系列，最终固定为主力模型。

**Why qwen3-32b instead of llama?**

Deep film profiles demand high-quality Chinese-English mixed writing, long-form structural consistency, and reliable citation format adherence. In practice, qwen3-32b significantly outperformed comparable llama models on Chinese writing quality and instruction following (particularly citation numbering), and was adopted as the primary model.

**DuckDuckGo 双兼容导入**：`ddgs` 与 `duckduckgo_search` 两个包名在不同部署环境下版本不一，代码层做了 try/except 双兼容导入以确保跨平台稳定性。

**DuckDuckGo dual-import compatibility**: The package name differs between `ddgs` and `duckduckgo_search` across deployment environments; a try/except dual-import handles both to ensure cross-platform stability.

---

*Independently designed and delivered · 2025–2026*
