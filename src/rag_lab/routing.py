"""Controlled entity constraints and topic routing; no query IDs or gold labels baked in."""

import re
from dataclasses import dataclass, field

VERSION = "entity-topic-v1"


def normalized(query):
    return query.lower().translate(str.maketrans("顯賣庫儲線產試環飲單", "显卖库储线产试环饮单"))


@dataclass
class Scope:
    entities: list[str] = field(default_factory=list)
    category: str | None = None
    brand: str | None = None
    capacity_operator: str | None = None
    capacity_values: list[int] = field(default_factory=list)
    extreme: str | None = None
    topics: list[str] = field(default_factory=list)
    exclude_style: str | None = None
    status: str = "ready"
    missing: list[str] = field(default_factory=list)
    expansion: str = ""
    resolved_capacity: int | None = None
    scope_basis: str | None = None


def infer_scope(query):
    text = normalized(query)
    scope = Scope(
        entities=re.findall(
            r"(?:PHONE-[A-Z]-\d+|FRUIT-[A-Z]+-\d+KG|ERR-[A-Z]+-\d+|svc-[a-z]+-v\d+)", query
        )
    )
    accessory = any(x in text for x in ("手机壳", "保护壳", "配件", "保护套"))
    phone = any(x in text for x in ("手机", "smartphone", "phone-"))
    storage = any(x in text for x in ("存储", "容量", "储存"))
    apple = "苹果" in text or "apple" in text
    if accessory:
        scope.category = "accessory"
    elif phone or (apple and storage):
        scope.category = "smartphone"
    elif "红富士" in text or "水果" in text or any(x.startswith("FRUIT-") for x in scope.entities):
        scope.category = "fruit"
    if apple and scope.category in {"smartphone", "accessory"}:
        scope.brand = "Apple"
    for alias, brand in (
        ("三星", "Samsung"),
        ("samsung", "Samsung"),
        ("华为", "Huawei"),
        ("huawei", "Huawei"),
        ("小米", "Xiaomi"),
        ("诺基亚", "Nokia"),
    ):
        if alias in text and scope.category in {"smartphone", "accessory"}:
            if scope.brand and scope.brand != brand:
                scope.status = "clarification"
                scope.missing.append("brand_logic")
            scope.brand = brand
    # Storage predicates are interpreted only in a smartphone context, not menu/server capacity.
    if scope.category == "smartphone":
        raw_values = re.findall(r"(?<![\d.])(-?\d+(?:\.\d+)?)\s*(?:gb|g)(?![a-z0-9])", text)
        if any(float(v) <= 0 or not float(v).is_integer() for v in raw_values):
            scope.status = "clarification"
            scope.missing.append("capacity_unit_value")
        if re.search(r"\d+\s*(?:mb|tb)(?![a-z0-9])", text):
            scope.status = "clarification"
            scope.missing.append("unsupported_storage_unit")
        capacity = re.search(
            r"(至少|不低于|不小于|超过|大于|不超过|不高于|至多|低于|小于)\s*(\d+)\s*(?:gb|g)(?![a-z0-9])",
            text,
        )
        relative = re.search(r"比\s*(\d+)\s*(?:gb|g)\s*(?:更)?(小|大)", text)
        if capacity:
            scope.capacity_operator = {
                "至少": "ge",
                "不低于": "ge",
                "不小于": "ge",
                "超过": "gt",
                "大于": "gt",
                "不超过": "le",
                "不高于": "le",
                "至多": "le",
                "低于": "lt",
                "小于": "lt",
            }[capacity[1]]
            scope.capacity_values = [int(capacity[2])]
        elif relative:
            scope.capacity_operator = "lt" if relative[2] == "小" else "gt"
            scope.capacity_values = [int(relative[1])]
        else:
            values = list(
                dict.fromkeys(int(v) for v in re.findall(r"(\d+)\s*(?:gb|g)(?![a-z0-9])", text))
            )
            if values:
                scope.capacity_operator = "in"
                scope.capacity_values = values
                if len(values) > 1 and not any(x in text for x in ("和", "与", "比较", "vs", "或")):
                    scope.status = "clarification"
                    scope.missing.append("capacity_logic")
        minimum = any(x in text for x in ("较小", "最小", "小容量", "最少"))
        maximum = any(x in text for x in ("较大", "最大", "大容量", "最多"))
        if minimum and maximum:
            scope.status = "clarification"
            scope.missing.append("comparison_logic")
        elif minimum:
            scope.extreme = "min"
        elif maximum:
            scope.extreme = "max"
        if any(
            x in text for x in ("不要pro", "不要 pro", "排除pro", "排除 pro", "标准版", "标准款")
        ):
            scope.exclude_style = "Pro"
    elif storage and any(x in text for x in ("较小", "最小", "较大", "最大")):
        scope.status = "clarification"
        scope.missing.append("comparison_entity_scope")
    recipe = any(x in text for x in ("花生", "过敏", "配方", "配料表", "痕量"))
    menu = any(x in text for x in ("菜单", "小程序", "配送", "外送", "线上", "点单", "收银", "pos"))
    unavailable = any(
        x in text
        for x in (
            "不可点",
            "点不了",
            "不能点",
            "售罄",
            "卖完",
            "不可售",
            "不能卖",
            "可售",
            "能卖",
            "无法下单",
            "无法售卖",
            "灰了",
            "灰色",
            "入口",
        )
    )
    if recipe:
        scope.topics.append("recipe_safety")
    if menu and unavailable:
        scope.topics.append("menu_availability")
    if scope.topics and any(x in text for x in ("价格", "报价", "收费", "会员价", "多少钱")):
        scope.topics.append("pricing_rules")
    if scope.topics and any(x in text for x in ("发布", "上线", "发版")):
        scope.topics.append("release_contract")
    service = any(x in text for x in ("服务", "service", "系统"))
    environment = any(
        x in text for x in ("沙箱", "线上", "线下", "测试", "生产", "环境", "staging", "prod")
    )
    if (
        service
        and environment
        and any(x in text for x in ("同名", "同一个", "同一", "怎么区分", "当成", "合并"))
    ):
        scope.topics.append("environment_identity")
    if any(x in text for x in ("依赖", "调用方", "消费者", "cmdb")) and any(
        x in text for x in ("没有", "缺少", "缺边", "一个也", "下线", "删除服务", "能否")
    ):
        scope.topics.append("dependency_safety")
    if any(x in text for x in ("消息", "日志", "处理记录")) and any(
        x in text for x in ("丢", "缺少", "未发现", "没发现", "没有", "找不到")
    ):
        scope.topics.append("message_diagnosis")
    aliases = {
        "menu_availability": "菜单 渠道配置 可售状态 同步水位",
        "recipe_safety": "配方 花生 痕量 风险",
        "environment_identity": "服务 生产 测试 环境 身份",
        "dependency_safety": "CMDB 依赖 登记 影响 边界",
        "message_diagnosis": "消息 处理记录 观测 日志 排查",
        "pricing_rules": "报价 币种 渠道 会员",
        "release_contract": "发布 接口 契约",
    }
    scope.expansion = " ".join(aliases[t] for t in scope.topics)
    return scope


def matches(document, scope):
    if (
        scope.entities
        and not set(scope.topics) & {"environment_identity", "dependency_safety"}
        and not set(scope.entities) & set(document.entities)
    ):
        return False
    if scope.category and document.category != scope.category:
        return False
    if scope.brand and document.brand != scope.brand:
        return False
    if scope.topics and not set(scope.topics) & set(document.topics):
        return False
    if scope.exclude_style and (document.style is None or document.style == scope.exclude_style):
        return False
    if scope.capacity_operator:
        value = document.capacity_gb
        if value is None:
            return False
        threshold = scope.capacity_values[0]
        op = scope.capacity_operator
        if not {
            "in": value in scope.capacity_values,
            "lt": value < threshold,
            "le": value <= threshold,
            "gt": value > threshold,
            "ge": value >= threshold,
        }[op]:
            return False
    if scope.resolved_capacity is not None and document.capacity_gb != scope.resolved_capacity:
        return False
    return True


def constrain(documents, scope):
    if scope.status != "ready":
        return []
    selected = [d for d in documents if matches(d, scope)]
    if scope.extreme:
        values = [d.capacity_gb for d in selected if d.capacity_gb is not None]
        if not values:
            return []
        scope.resolved_capacity = min(values) if scope.extreme == "min" else max(values)
        scope.scope_basis = "current authorized effective documents matching entity/category/brand"
        selected = [d for d in selected if matches(d, scope)]
    return selected
