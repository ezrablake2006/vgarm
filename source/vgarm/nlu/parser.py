import re
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class Intent:
    kind: Literal[
        "move_to_dir",
        "lift",
        "move_next_to_object",
        "swap",
        "place_to_center",
    ]
    object_color: Optional[str] = None
    object_category: Optional[str] = None
    target_keyword: Optional[str] = None
    ref_color: Optional[str] = None


_COLOR_MAP: dict[str, str] = {
    "红色": "red",
    "红": "red",
    "黄色": "yellow",
    "黄": "yellow",
    "蓝色": "blue",
    "蓝": "blue",
    "绿色": "green",
    "绿": "green",
    "白色": "white",
    "白": "white",
    "黑色": "black",
    "黑": "black",
}

_CATEGORY_MAP: dict[str, str] = {
    "方块": "cube",
    "立方体": "cube",
    "盒子": "box",
    "箱子": "box",
    "杯子": "cup",
    "瓶子": "bottle",
    "工具": "tool",
    "物体": "object",
}

_TARGETS = [
    "左边",
    "右边",
    "前面",
    "后面",
    "中间",
    "左侧",
    "右侧",
    "前方",
    "后方",
    "中央",
]


def _norm_target(t: str) -> str:
    if t in ("左侧",):
        return "左边"
    if t in ("右侧",):
        return "右边"
    if t in ("前方",):
        return "前面"
    if t in ("后方",):
        return "后面"
    if t in ("中央",):
        return "中间"
    return t


def parse_cn(text: str) -> Intent:
    t = re.sub(r"\s+", "", text)
    if not t:
        raise ValueError("empty command")

    ms2 = re.search(r"(把|将)(?P<obj>.+?)放在(?P<ref>.+?)(的)?(?P<dir>左边|右边|前面|后面|左侧|右侧|前方|后方)$", t)
    if ms2:
        obj = ms2.group("obj")
        ref = ms2.group("ref")
        dirw = _norm_target(ms2.group("dir"))
        color = None
        for k, v in _COLOR_MAP.items():
            if k in obj:
                color = v
                break
        category = None
        for k, v in _CATEGORY_MAP.items():
            if k in obj:
                category = v
                break
        ref_color = None
        for k, v in _COLOR_MAP.items():
            if k in ref:
                ref_color = v
                break
        return Intent(kind="move_next_to_object", object_color=color, object_category=category, target_keyword=dirw, ref_color=ref_color)

    m = re.search(r"(把|将)(?P<obj>.+?)(移到|放到|放在|搬到)(?P<dst>.+)$", t)
    if not m:
        if re.search(r"(把|将).+(抬起|提起|拿起|抓起)$", t):
            obj = re.sub(r"^(把|将)|(抬起|提起|拿起|抓起)$", "", t)
            color = None
            for k, v in _COLOR_MAP.items():
                if k in obj:
                    color = v
                    break
            category = None
            for k, v in _CATEGORY_MAP.items():
                if k in obj:
                    category = v
                    break
            return Intent(kind="lift", object_color=color, object_category=category)
        if re.search(r"(把|将).+(放回原位|放回原地)$", t):
            obj = re.sub(r"^(把|将)|(放回原位|放回原地)$", "", t)
            color = None
            for k, v in _COLOR_MAP.items():
                if k in obj:
                    color = v
                    break
            category = None
            for k, v in _CATEGORY_MAP.items():
                if k in obj:
                    category = v
                    break
            return Intent(kind="place_to_center", object_color=color, object_category=category)
        ms = re.search(r"交换(?P<a>.+)和(?P<b>.+)位置", t)
        if not ms:
            ms = re.search(r"将(?P<a>.+)和(?P<b>.+)互换位置", t)
        if ms:
            a = ms.group("a")
            b = ms.group("b")
            def pick_color(s: str) -> Optional[str]:
                for k, v in _COLOR_MAP.items():
                    if k in s:
                        return v
                return None
            return Intent(kind="swap", object_color=pick_color(a), ref_color=pick_color(b))
        raise ValueError(f"unsupported command: {text}")

    obj = m.group("obj")
    dst = m.group("dst")

    color = None
    for k, v in _COLOR_MAP.items():
        if k in obj:
            color = v
            break

    category = None
    for k, v in _CATEGORY_MAP.items():
        if k in obj:
            category = v
            break

    target = None
    for kw in _TARGETS:
        if kw in dst:
            target = kw
            break

    if target is None:
        target = dst
    target = _norm_target(target)

    return Intent(kind="move_to_dir", object_color=color, object_category=category, target_keyword=target)
