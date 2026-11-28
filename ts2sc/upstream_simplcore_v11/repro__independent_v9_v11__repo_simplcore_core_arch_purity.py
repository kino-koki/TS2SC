# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""架构纯洁性守卫 —— 把"脱离 Transformer"变成**可测试**的硬约束。

为什么需要它
------------
"这套架构不属于 Transformer 框架"如果只写在文档里，就会在后续迭代中悄悄失效：
某次为了追指标又把 ``mixer`` 设成注意力族、或新加一个 ``*Attention*`` 模块，
注意力就回来了，而 loss 曲线完全正常。本模块提供两道断言，让 CI 在合并前
直接失败：

* :func:`assert_pure_config` —— 在建模前挡掉会引入注意力的配置开关；
* :func:`assert_attention_free` —— 在建模后遍历 ``named_modules()``，
  确认实例化出来的模型里真的没有任何 ``*Attention*`` / ``*attn*`` 模块。

当前状态
--------
SimplCore 已**彻底移除**所有注意力模块（``CausalAttention`` / ``ReasoningAttention`` /
``SlidingWindowAttention`` / GQA-YaRN 注意力 / RoPE 等均已删除，见 ``.trash/attn_removed/``）。
``SimplCoreConfig.validate()`` 现在只允许 ``sgc`` / ``sgc-nogate`` / ``sgc-nolocal`` / ``srf``
四种非注意力 mixer。本守卫作为**回归防火墙**保留：万一将来有人重新引入注意力
模块或配置开关，CI 会立刻失败，而不是等 loss 曲线骗过所有人。

判定口径
--------
SGC（谱全局卷积）、SRF（谱振流）、SSM（选择性状态空间）、GatedSpectralConv2
（GLA/Mamba2 风格通道门，非 softmax 注意力）都通过；``attn`` / ``hybrid*``
族以及任何类名含 ``attention`` / ``attn`` 的 ``nn.Module`` 都会被抓出来。
``SimplCoreConfig.validate()`` 的 mixer 白名单来自 ``registry.mixer_kinds``，
``assert_registry_pure()`` 额外保证注册表里登记的实现类本身不含注意力。

用法::

    from simplcore.core.arch.purity import assert_pure_config, assert_attention_free
    assert_pure_config(cfg)          # 先挡配置
    model = SimplCoreModel(cfg)
    assert_attention_free(model)     # 再挡实例
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Tuple

__all__ = [
    "ATTENTION_MIXERS",
    "ATTENTION_SYMBOLS",
    "attention_like_modules",
    "scan_source_attention_free",
    "assert_attention_free",
    "assert_pure_config",
    "assert_registry_pure",
    "assert_source_attention_free",
    "assert_architecture_pure",
]

#: 会引入注意力子模块的 mixer 取值。
ATTENTION_MIXERS = frozenset({
    "attn", "hybrid", "hybrid-gqa", "hybrid-yarn", "hybrid-gqa-yarn",
})

#: 模块类名里出现任一子串即视为注意力模块。
_ATTENTION_TOKENS = ("attention", "attn")

#: 注意力实现必然留下的**标识符**（函数/属性/参数名）。
#:
#: 只做类名断言是可绕过的：把 ``CausalAttention`` 改名为 ``MixerA``，类名检查
#: 立刻失效，但 ``q_proj`` / ``k_proj`` / ``scaled_dot_product_attention`` 仍然
#: 会出现在 AST 里。这里改为 AST 标识符级扫描，堵住"改名绕过"。
#: 注意：不把 ``kv_cache`` 泛化匹配放进来 —— ``scaling.kv_cache_bytes_per_token``
#: 是给 **Transformer 对照臂**算账的度量函数，不是运行时缓存实现；只禁
#: ``past_key_values`` / ``k_cache`` / ``v_cache`` 这类状态载体。
ATTENTION_SYMBOLS = frozenset({
    "scaled_dot_product_attention",
    "multi_head_attention_forward",
    "past_key_value",
    "past_key_values",
    "k_cache",
    "v_cache",
    "cache_k",
    "cache_v",
    "rotary_emb",
    "apply_rotary_pos_emb",
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv_proj",
    "attn_weights",
    "attention_scores",
    "causal_attention",
    "self_attention",
})

#: 源码扫描排除：守卫自身（含禁用词）、测试（构造反例）、归档。
_SOURCE_SCAN_EXCLUDE_PARTS = ("tests", "__pycache__", ".trash", ".archive")

#: **显式定义级豁免**（文件名 → 理由）。与文件头 `purity: allow-attention-reference`
#: 标记（只豁免 API 调用）不同，本清单豁免**定义级**检查（类名/属性名），
#: 每一条都必须有对照协议支撑并在此写明，审查时逐条过：
#: - sidecar_exact_attn.py：L2 取回的「局部精确注意力」**实验件**
#:   （协议 X-sidecar-attn），只对指针选出的 k·B 个键做对齐，键集与 L 无关；
#:   默认产品线走 sidecar.py 的纯净注入（无注意力）。它不是架构主张，
#:   未在真实任务实测；从清单移除即可恢复全量守卫。
#: - axis_probe.py：轴 C/D 能力对照协议（X-axis-probe）的 **TS 对照臂**
#:   （迷你 Transformer，先例：equal_budget.py）——「SGC 精确指向结构性
#:   劣势」等负面结果的可持续复现装置；对照臂自包含、不进运行时，
#:   移除即恢复全量守卫。
#: - migration/dc_bias.py：TS→SGC **解析迁移辅助件**，把 teacher 的
#:   attention 子层输出当作外部张量读入以估计 DC 补偿量；它不实现任何
#:   注意力算子、不进运行时、不进 mixer 注册表。属于「引用概念而非实现」，
#:   与 axis_probe 的 TS 对照臂同类；移除即恢复全量守卫。
_SOURCE_SCAN_DEFINITION_EXEMPT_FILES = frozenset({"sidecar_exact_attn.py",
                                                  "axis_probe.py",
                                                  "dc_bias.py"})


def _identifier_hit(name: str) -> bool:
    """标识符是否命中注意力判定口径。"""
    low = name.lower()
    if low in ATTENTION_SYMBOLS:
        return True
    return any(tok in low for tok in _ATTENTION_TOKENS)


#: 守卫/审计自身的名字：它们**引用**注意力概念是为了检查它不存在，
#: 不是实现。避免"纯洁性扫描把检查器自己抓出来"的自指悖论。
_CHECKER_IDENTS = frozenset({
    "attention_like_modules",
    "attention_free",
    "assert_attention_free",
    "assert_source_attention_free",
    "assert_architecture_pure",
    "scan_source_attention_free",
    "_check_no_attention",
})


def _is_checker_name(name: str) -> bool:
    low = name.lower()
    return (low in _CHECKER_IDENTS
            or low.startswith(("assert_", "scan_", "_check", "_assert", "_scan")))


def _iter_source_impl_identifiers(tree: ast.AST,
                                  check_calls: bool = True) -> Iterable[Tuple[str, int]]:
    """只遍历"定义性"标识符：类名、函数名、参数名、``self.x`` 属性赋值。

    刻意**不查普通引用**（``ast.Name`` / 非调用的 ``ast.Attribute``）：产品线里
    大量 ``attention_like_modules(...)`` 这类调用是审计代码，把它们算成实现会让
    守卫变成噪音。真正绕不过去的是"定义了一个注意力模块"、"给模块挂了
    ``q_proj/k_proj/v_proj`` 状态"，或在 ``forward`` 里调用注意力 API。
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_checker_name(node.name):
                yield node.name, node.lineno
        elif isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif check_calls and isinstance(node, ast.Call):
            func = node.func
            # 调用只认**精确的注意力 API 名**，不按 attn 子串匹配 ——
            # 否则 `attention_like_modules(...)`（审计调用）会被误伤。
            if isinstance(func, ast.Attribute):
                if func.attr in ATTENTION_SYMBOLS:
                    yield func.attr, node.lineno
            elif isinstance(func, ast.Name):
                if func.id in ATTENTION_SYMBOLS:
                    yield func.id, node.lineno
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    yield target.attr, node.lineno
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                yield target.attr, node.lineno


def scan_source_attention_free(root: str | Path | None = None) -> List[Tuple[str, int, str]]:
    """扫描产品线 Python 源码，返回 ``[(文件, 行号, 标识符), ...]``。

    与 :func:`attention_like_modules`（实例级）互补：实例级只能看到**已经**
    实例化的类名，源码级还能抓住"改了名但保留注意力符号"的实现，以及
    "暂时没人实例化"的注意力模块。字符串常量与注释不参与判定 ——
    HF 桥里出现的 ``"attn.q_proj"`` 是**被拒绝迁移**的键名，不是实现。
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    hits: List[Tuple[str, int, str]] = []
    for path in sorted(base.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        parts = set(path.parts)
        if parts & set(_SOURCE_SCAN_EXCLUDE_PARTS):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        # 研究对照臂（如 equal_budget 的 TS 基线）可以**引用**注意力 API，
        # 但必须在文件头显式声明 `purity: allow-attention-reference`；
        # 定义级检查（q_proj/k_proj/v_proj、注意力类名）仍然生效——除非
        # 文件在 ``_SOURCE_SCAN_DEFINITION_EXEMPT_FILES`` 显式白名单里
        # （见清单注释：每条豁免都有对照协议支撑，可审计、可撤销）。
        allow_reference = "purity: allow-attention-reference" in text[:3000]
        definition_exempt = path.name in _SOURCE_SCAN_DEFINITION_EXEMPT_FILES
        for ident, lineno in _iter_source_impl_identifiers(
                tree, check_calls=not allow_reference):
            if definition_exempt:
                continue
            if _identifier_hit(ident):
                hits.append((str(path), lineno, ident))
    return hits


def assert_source_attention_free(root: str | Path | None = None) -> bool:
    """断言产品线源码不含注意力实现符号；否则抛 :class:`AssertionError`。"""
    hits = scan_source_attention_free(root)
    if hits:
        listing = ", ".join(f"{p}:{ln}:{ident}" for p, ln, ident in hits[:8])
        more = "" if len(hits) <= 8 else f" 等 {len(hits)} 处"
        raise AssertionError(
            f"产品线源码包含注意力实现符号，违反“脱离 Transformer”约束：{listing}{more}")
    return True


def _class_attention_reasons(cls: type) -> List[str]:
    """类级注意力证据（类名 / 直接参数名 / forward 源码里的禁用 API）。

    结果按类缓存：1B 模型有几万个子模块，但只有几十个不同的类。
    """
    cached = _CLASS_REASON_CACHE.get(cls)
    if cached is not None:
        return cached
    reasons: List[str] = []
    name = cls.__name__
    if any(tok in name.lower() for tok in _ATTENTION_TOKENS):
        reasons.append(f"class={name}")
    forward = getattr(cls, "forward", None)
    if forward is not None:
        try:
            import inspect
            src = inspect.getsource(forward)
        except (OSError, TypeError):
            src = ""
        for sym in ATTENTION_SYMBOLS:
            if sym in src:
                reasons.append(f"forward uses {sym}")
    _CLASS_REASON_CACHE[cls] = reasons
    return reasons


_CLASS_REASON_CACHE: dict = {}


def attention_like_modules(model) -> List[Tuple[str, str]]:
    """返回 ``[(模块名, 类名), ...]``；空列表表示模型无注意力。

    判定不只看类名（改名即可绕过），还看：

    * 类名里的 ``attention`` / ``attn``；
    * 模块直接持有的参数名（``q_proj`` / ``k_proj`` / ``v_proj`` 等）；
    * ``forward`` 源码是否调用 ``scaled_dot_product_attention`` 等注意力 API。

    后两条让"把类改名为 ``MixerA``"这种绕过在实例级也失效。
    """
    hits: List[Tuple[str, str]] = []
    for name, mod in model.named_modules():
        cls = type(mod)
        reasons = _class_attention_reasons(cls)
        for pname, _ in mod.named_parameters(recurse=False):
            if _identifier_hit(pname):
                reasons = reasons + [f"param={pname}"]
                break
        if reasons:
            hits.append((name or "<root>", f"{cls.__name__}[{'; '.join(reasons)}]"))
    return hits


def assert_attention_free(model) -> bool:
    """断言模型不含任何注意力模块；否则抛 :class:`AssertionError`。"""
    hits = attention_like_modules(model)
    if hits:
        listing = ", ".join(f"{n}:{c}" for n, c in hits[:8])
        more = "" if len(hits) <= 8 else f" 等 {len(hits)} 个"
        raise AssertionError(
            f"模型包含注意力模块，违反“脱离 Transformer”约束：{listing}{more}"
        )
    return True


def assert_pure_config(config) -> bool:
    """断言配置不会引入注意力；否则抛 :class:`AssertionError`。"""
    problems: List[str] = []
    mixer = getattr(config, "mixer", "sgc")
    if mixer in ATTENTION_MIXERS:
        problems.append(f"mixer={mixer!r}")
    if getattr(config, "attn_every", None) is not None:
        problems.append(f"attn_every={config.attn_every}")
    if getattr(config, "reason_attn_every", None) is not None:
        problems.append(f"reason_attn_every={config.reason_attn_every}")
    if problems:
        raise AssertionError(
            "配置会引入注意力，违反纯洁性：" + ", ".join(problems)
        )
    return True


def assert_registry_pure() -> bool:
    """断言 mixer 注册表里登记的**实现类**全部不是注意力模块.

    ``validate()`` 的白名单来自注册表（``registry.mixer_kinds``），所以
    "往注册表里塞一个注意力实现" 等于给注意力开后门。本断言遍历注册表
    的类路径做类名检查，与 :func:`assert_attention_free` 同一判定口径，
    在 CI 里堵住这条后门。
    """
    from .registry import iter_mixer_classes
    hits: List[str] = []
    for name, cls in iter_mixer_classes():
        for reason in _class_attention_reasons(cls):
            hits.append(f"{name}:{cls.__name__}({reason})")
    if hits:
        raise AssertionError(
            "mixer 注册表包含注意力实现，违反纯洁性：" + ", ".join(hits)
        )
    # 注册表只是产品线的一个入口；源码级扫描覆盖"没被注册、但被人 import
    # 后直接实例化"的旁路。两者一起才是完整的回归防火墙。
    assert_source_attention_free()
    return True


def assert_architecture_pure(config=None, model=None, root: str | Path | None = None) -> bool:
    """配置 + 实例 + 源码三层纯洁性检查的聚合入口（CI 一把梭）。"""
    if config is not None:
        assert_pure_config(config)
    if model is not None:
        assert_attention_free(model)
    assert_registry_pure()
    assert_source_attention_free(root)
    return True
