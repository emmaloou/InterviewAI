import ast
import json
import re
from typing import Any

try:
    import json5
except ImportError:  # pragma: no cover
    json5 = None


_TOKEN_FIXER = re.compile(
    r'(?<![A-Za-z0-9_"\'])\b(true|false|null)\b(?![A-Za-z0-9_"\'])',
    re.IGNORECASE,
)


def _harmonize_brackets(text: str) -> str:
    """Corrige l'ordre de fermeture des crochets/accolades en ignorant le contenu des chaînes."""
    if not text:
        return text

    result = []
    stack = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            result.append(char)
            continue

        if char in "{[":
            stack.append(char)
            result.append(char)
            continue

        if char in "}]":
            if not stack:
                # ignorer le caractère orphelin
                continue

            expected = stack[-1]
            if (expected == "{" and char == "}") or (expected == "[" and char == "]"):
                stack.pop()
                result.append(char)
                continue

            # si la fermeture ne correspond pas, substituer par celle attendue
            stack.pop()
            substitute = "]" if expected == "[" else "}"
            result.append(substitute)
            continue

        result.append(char)

    # Ajouter les fermetures manquantes restantes
    while stack:
        opener = stack.pop()
        result.append("}" if opener == "{" else "]")

    return "".join(result)


def _balance_brackets(text: str) -> str:
    """Nettoie approximativement le JSON retourné par un LLM."""
    balanced = (text or "").strip()
    if not balanced:
        return balanced

    # Supprimer les caractères après la dernière accolade/crochet fermant
    last_idx = max(balanced.rfind("}"), balanced.rfind("]"))
    if last_idx != -1:
        balanced = balanced[: last_idx + 1]

    # Supprimer les virgules finales avant ] ou }
    balanced = re.sub(r",\s*(\]|\})", r"\1", balanced)

    # Supprimer les virgules en fin de document
    balanced = balanced.rstrip(",")

    for open_bracket, close_bracket in (("{", "}"), ("[", "]")):
        diff = balanced.count(open_bracket) - balanced.count(close_bracket)
        if diff > 0:
            balanced += close_bracket * diff
        elif diff < 0:
            # trop de closings: retirer les derniers
            remove_count = abs(diff)
            while remove_count > 0 and balanced:
                if balanced.endswith(close_bracket):
                    balanced = balanced[:-1]
                    remove_count -= 1
                else:
                    break
    return balanced


def _normalize_literals(text: str) -> str:
    """Convertit true/false/null en True/False/None hors des chaînes."""
    return _TOKEN_FIXER.sub(
        lambda match: {"true": "True", "false": "False", "null": "None"}[
            match.group(1).lower()
        ],
        text,
    )


def safe_json_loads(payload: str) -> Any:
    """Parse une chaîne JSON en tolérant plusieurs formats imparfaits."""
    if payload is None:
        raise ValueError("Payload JSON vide.")

    if isinstance(payload, (dict, list)):
        return payload

    harmonized = _harmonize_brackets(payload)
    candidate = _balance_brackets(harmonized)
    last_error = None

    for strict in (True, False):
        try:
            return json.loads(candidate, strict=strict)
        except json.JSONDecodeError as exc:
            last_error = exc

    if json5:
        try:
            return json5.loads(candidate)
        except Exception as exc:  # pragma: no cover
            last_error = exc

    try:
        python_like = _normalize_literals(candidate)
        return ast.literal_eval(python_like)
    except Exception as exc:  # pragma: no cover
        last_error = exc

    raise ValueError(f"Impossible de parser le JSON: {last_error}")

