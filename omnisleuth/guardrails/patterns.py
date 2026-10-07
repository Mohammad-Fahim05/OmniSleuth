"""Deterministic pattern definitions and regex rules for guardrails."""

import re
from typing import List, Pattern, Tuple

# 1. Prompt Injection Detection Patterns (Normalized case-insensitive regex)
PROMPT_INJECTION_PATTERNS: List[Tuple[str, Pattern]] = [
    ("ignore_previous_instructions", re.compile(r"\b(?:ignore|forget|disregard|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|rules|prompts?)\b", re.IGNORECASE)),
    ("reveal_system_prompt", re.compile(r"\b(?:reveal|show|print|display|dump|leak|tell\s+me)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions|system\s+rules|developer\s+prompt|internal\s+prompt)\b", re.IGNORECASE)),
    ("forget_system_rules", re.compile(r"\b(?:forget|disregard|override|bypass|ignore)\s+(?:all\s+)?(?:system\s+rules|rules|restrictions|guardrails|safety\s+filters|limitations)\b", re.IGNORECASE)),
    ("role_reassignment", re.compile(r"\b(?:you\s+are\s+no\s+longer|stop\s+acting\s+as)\s+(?:omnisleuth|an\s+investigation\s+system|a\s+data\s+assistant)\b", re.IGNORECASE)),
    ("disregard_tool_restrictions", re.compile(r"\b(?:disregard|ignore)\s+(?:the\s+)?(?:tool\s+restrictions|allowlisted\s+tools|sandbox)\b", re.IGNORECASE)),
    ("obey_user_only", re.compile(r"\b(?:obey\s+me|obey\s+user|follow\s+only\s+me|obey\s+user\s+only)\b", re.IGNORECASE)),
]

# 2. Jailbreak Detection Patterns
JAILBREAK_PATTERNS: List[Tuple[str, Pattern]] = [
    ("developer_mode", re.compile(r"\b(?:enter|activate|switch\s+to|enable)\s+(?:developer\s+mode|dan\s+mode|unfiltered\s+mode|god\s+mode)\b", re.IGNORECASE)),
    ("act_unrestricted", re.compile(r"\b(?:act|pretend|behave)\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|jailbroken|unaligned)\s+(?:ai|assistant|model|bot)\b", re.IGNORECASE)),
    ("pretend_no_restrictions", re.compile(r"\bpretend\s+(?:you\s+have\s+no|there\s+are\s+no)\s+(?:restrictions|boundaries|rules|limits)\b", re.IGNORECASE)),
    ("disable_safety", re.compile(r"\b(?:disable|turn\s+off|deactivate)\s+(?:your\s+)?(?:safety|guardrails|filters|content\s+moderation)\b", re.IGNORECASE)),
    ("ignore_limitations", re.compile(r"\b(?:ignore|remove)\s+(?:your\s+)?(?:limitations|ethical\s+guidelines|constraints)\b", re.IGNORECASE)),
]

# 3. Arbitrary Code Execution Request Patterns
ARBITRARY_CODE_PATTERNS: List[Tuple[str, Pattern]] = [
    ("eval_exec_call", re.compile(r"\b(?:exec|eval)\s*\(", re.IGNORECASE)),
    ("os_subprocess_call", re.compile(r"\b(?:os\.system|subprocess\.(?:run|Popen|call)|os\.popen)\b", re.IGNORECASE)),
    ("shell_spawn", re.compile(r"\b(?:powershell(?:\.exe)?|cmd\.exe|/bin/bash|/bin/sh)\b", re.IGNORECASE)),
    ("direct_execution_request", re.compile(r"\b(?:run|execute|generate\s+and\s+execute)\s+(?:this\s+)?(?:arbitrary\s+)?(?:python\s+code|bash\s+script|shell\s+command|code\s+block)\b", re.IGNORECASE)),
    ("open_shell", re.compile(r"\b(?:open|start)\s+(?:a\s+)?(?:shell|terminal|command\s+prompt)\b", re.IGNORECASE)),
]

# 4. Off-Topic Query Indicators
OFF_TOPIC_PATTERNS: List[Tuple[str, Pattern]] = [
    ("creative_writing", re.compile(r"\b(?:write|compose|generate|tell\s+me)\s+(?:me\s+)?(?:a\s+)?(?:funny\s+)?(?:poem|poetry|song|story|novel|joke|haiku|riddle|rap|essay)\b", re.IGNORECASE)),
    ("weather_inquiry", re.compile(r"\b(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:today(?:'s)?\s+)?weather|will\s+it\s+rain|temperature\s+outside|forecast\s+for\s+tomorrow)\b", re.IGNORECASE)),
    ("general_trivia_politics", re.compile(r"\b(?:who\s+is\s+the\s+(?:current\s+)?(?:president|prime\s+minister|king|queen)|capital\s+of\s+[a-zA-Z]+|who\s+won\s+the\s+(?:cricket|football|world\s+cup|match))\b", re.IGNORECASE)),
    ("unrelated_science", re.compile(r"\b(?:solve\s+this\s+(?:physics|chemistry|calculus|math)[\w\s]*(?:problem|equation|homework)|calculate\s+gravity\s+on\s+mars)\b", re.IGNORECASE)),
    ("personal_advice", re.compile(r"\b(?:diagnose\s+my\s+symptoms|medical\s+advice|dating\s+advice|horoscope)\b", re.IGNORECASE)),
]

# On-Topic Domain Vocabulary (Boosts classification into analytical domain)
ON_TOPIC_KEYWORDS = {
    "revenue", "sales", "order", "orders", "aov", "category", "region", "trend",
    "decline", "increase", "drop", "growth", "margin", "contribution", "segment",
    "decomposition", "anomaly", "anomalies", "outlier", "outliers", "customer",
    "dataset", "data", "metric", "metrics", "kpi", "report", "quarterly", "march",
    "stockout", "supplier", "refund", "return", "product", "column", "units",
    "quantity", "volume", "inventory", "summary", "breakdown", "distribution"
}

# 5. Secret Leakage Detection Patterns (Redaction & Blocking)
SECRET_LEAKAGE_PATTERNS: List[Tuple[str, Pattern]] = [
    ("openai_api_key", re.compile(r"sk-(?:proj-)?[a-zA-Z0-9_\-]{20,}", re.IGNORECASE)),
    ("bearer_token", re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE)),
    ("generic_api_key", re.compile(r"(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}['\"]?", re.IGNORECASE)),
    ("password_assignment", re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{4,}['\"]", re.IGNORECASE)),
    ("private_key_header", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE)),
]

# 6. Unsupported Causal Phrases (Rule 6 & 7) - Ordered from longest/most specific to general
UNSUPPORTED_CAUSAL_PHRASES: List[Tuple[str, str]] = [
    (r"\bwas caused by\b", "was associated with"),
    (r"\bdirectly caused\b", "strongly contributed to"),
    (r"\bcaused the\b", "contributed to the"),
    (r"\bdefinitely resulted from\b", "likely coincided with"),
    (r"\bguaranteed cause\b", "observed contributor"),
    (r"\bthe sole reason\b", "a primary observed factor"),
    (r"\bproves that\b", "suggests that"),
    (r"\bcaused\b", "contributed to"),
    (r"\bled to\b", "was associated with"),
    (r"\bbecause of\b", "coinciding with"),
    (r"\bresulted from\b", "followed"),
]
