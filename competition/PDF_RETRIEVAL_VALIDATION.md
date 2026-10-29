# PDF 页面证据验证

实现入口为 `powercontext.sources.pdf_pages`。解析使用完整原 PDF 的全部物理页；Source 和标准文本投影保存完整正文。
检索仅使用 scoped Memory 的公开搜索接口，按词项、财务报表标题、业务章节、文档名及年度重排实际返回的候选。
实现不读取答案、证据标注、预解析正文或评测资产。

长文档名的页身份使用文档名 SHA-256、原 PDF SHA-256 和物理页码，避免超过 Source 的 256 字符上限。
普通长度文档名保持原有百分号编码身份。

有 canonical Source 引用的 `source-evidence` 条目保存完整证据正文，不受普通 Memory 笔记的 8192 字节上限限制。
写入、内容哈希校验及历史版本读取使用相同规则。没有提交 Source 证据的显式笔记和修改仍执行原有上限。
完整证据入库与回读在 SQLite 上验证，其他后端的存储容量仍然适用。

## 已执行的检查

在用户提供的 CentOS 虚拟机上运行无网络 Docker 自测容器，使用公开环境的 Python 3.12、pypdf 6.19.0 和固定依赖。

- PDF 公开验收：6 项通过，完整解析并持久化 PepsiCo 原 PDF 的 503 页，检查 5 个问题及重启后的精确引用。
- 新增 Runtime 回归：2 项通过，覆盖超过 8192 字节的完整页面、空白页、长 Unicode 文档名、重复提交、重启读取、
  多文档报表排序及正文修改的原有上限。
- Memory canonical 和 service 检查：18 项通过。
- 上述最终组合：26 项通过，耗时 57.09 秒。
- CSV、Pi 公开验收及相关 Memory 回归的兼容检查通过。
- 修改的 Python 文件通过 Ruff 检查和格式检查；`git diff --check` 通过。

组合检查命令（已准备固定依赖环境）：

```bash
PYTHONPATH=src:competition python -m pytest -q \
  -c competition/pytest.ini --confcutdir=competition \
  tests/e2e/test_pdf_page_evidence.py \
  tests/builtin/artifacts/memory/test_canonical.py \
  tests/builtin/artifacts/memory/test_service.py \
  competition/challenges/source-03-pdf-retrieval/acceptance
```

本机完整预提交检查因离线缓存缺少依赖而未完成。公开自测不提供质量 gold；以上结果不代表已测得正式的
证据词项召回、标注整页 F1 或公开及留出组的 evidence-page Recall@20，也不能据此保证正式得分。
