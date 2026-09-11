"""Identity validation shared by persisted paths and runtime preparation."""


def validate_conversation_id(conversation_id: str) -> None:
    if (
        not isinstance(conversation_id, str)
        or not conversation_id
        or len(conversation_id) > 80
        or "/" in conversation_id
        or "\\" in conversation_id
        or conversation_id in {".", ".."}
        or any(ord(char) < 32 or ord(char) == 127 for char in conversation_id)
    ):
        raise ValueError("invalid conversation ID")
