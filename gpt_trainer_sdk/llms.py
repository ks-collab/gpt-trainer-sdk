from enum import Enum
from typing import Dict, Union


class ModelType(str, Enum):
    """Enum for all valid model options supported by GPT Trainer SDK"""

    # Claude 3 Haiku models
    CLAUDE_3_HAIKU_4K = "claude-3-haiku-4k"
    CLAUDE_3_HAIKU_8K = "claude-3-haiku-8k"
    CLAUDE_3_HAIKU_16K = "claude-3-haiku-16k"
    CLAUDE_3_HAIKU_32K = "claude-3-haiku-32k"
    CLAUDE_3_HAIKU_64K = "claude-3-haiku-64k"
    CLAUDE_3_HAIKU_128K = "claude-3-haiku-128k"

    # Claude 3 Opus models
    CLAUDE_3_OPUS_2K = "claude-3-opus-2k"
    CLAUDE_3_OPUS_4K = "claude-3-opus-4k"
    CLAUDE_3_OPUS_8K = "claude-3-opus-8k"
    CLAUDE_3_OPUS_16K = "claude-3-opus-16k"
    CLAUDE_3_OPUS_32K = "claude-3-opus-32k"
    CLAUDE_3_OPUS_64K = "claude-3-opus-64k"
    CLAUDE_3_OPUS_128K = "claude-3-opus-128k"

    # Claude 3.5 Haiku models
    CLAUDE_3_5_HAIKU_2K = "claude-3.5-haiku-2k"
    CLAUDE_3_5_HAIKU_4K = "claude-3.5-haiku-4k"
    CLAUDE_3_5_HAIKU_8K = "claude-3.5-haiku-8k"
    CLAUDE_3_5_HAIKU_16K = "claude-3.5-haiku-16k"
    CLAUDE_3_5_HAIKU_32K = "claude-3.5-haiku-32k"
    CLAUDE_3_5_HAIKU_64K = "claude-3.5-haiku-64k"
    CLAUDE_3_5_HAIKU_128K = "claude-3.5-haiku-128k"

    # Claude 3.5 Sonnet models
    CLAUDE_3_5_SONNET_2K = "claude-3.5-sonnet-2k"
    CLAUDE_3_5_SONNET_4K = "claude-3.5-sonnet-4k"
    CLAUDE_3_5_SONNET_8K = "claude-3.5-sonnet-8k"
    CLAUDE_3_5_SONNET_16K = "claude-3.5-sonnet-16k"
    CLAUDE_3_5_SONNET_32K = "claude-3.5-sonnet-32k"
    CLAUDE_3_5_SONNET_64K = "claude-3.5-sonnet-64k"
    CLAUDE_3_5_SONNET_128K = "claude-3.5-sonnet-128k"

    # Claude 3.7 Sonnet models
    CLAUDE_3_7_SONNET_2K = "claude-3.7-sonnet-2k"
    CLAUDE_3_7_SONNET_4K = "claude-3.7-sonnet-4k"
    CLAUDE_3_7_SONNET_8K = "claude-3.7-sonnet-8k"
    CLAUDE_3_7_SONNET_16K = "claude-3.7-sonnet-16k"
    CLAUDE_3_7_SONNET_32K = "claude-3.7-sonnet-32k"
    CLAUDE_3_7_SONNET_64K = "claude-3.7-sonnet-64k"
    CLAUDE_3_7_SONNET_128K = "claude-3.7-sonnet-128k"
    CLAUDE_3_7_SONNET_200K = "claude-3.7-sonnet-200k"

    # Claude 4.0 Opus models
    CLAUDE_4_0_OPUS_2K = "claude-4.0-opus-2k"
    CLAUDE_4_0_OPUS_4K = "claude-4.0-opus-4k"
    CLAUDE_4_0_OPUS_8K = "claude-4.0-opus-8k"
    CLAUDE_4_0_OPUS_16K = "claude-4.0-opus-16k"
    CLAUDE_4_0_OPUS_32K = "claude-4.0-opus-32k"
    CLAUDE_4_0_OPUS_64K = "claude-4.0-opus-64k"
    CLAUDE_4_0_OPUS_128K = "claude-4.0-opus-128k"
    CLAUDE_4_0_OPUS_200K = "claude-4.0-opus-200k"

    # Claude 4.0 Sonnet models
    CLAUDE_4_0_SONNET_2K = "claude-4.0-sonnet-2k"
    CLAUDE_4_0_SONNET_4K = "claude-4.0-sonnet-4k"
    CLAUDE_4_0_SONNET_8K = "claude-4.0-sonnet-8k"
    CLAUDE_4_0_SONNET_16K = "claude-4.0-sonnet-16k"
    CLAUDE_4_0_SONNET_32K = "claude-4.0-sonnet-32k"
    CLAUDE_4_0_SONNET_64K = "claude-4.0-sonnet-64k"
    CLAUDE_4_0_SONNET_128K = "claude-4.0-sonnet-128k"
    CLAUDE_4_0_SONNET_200K = "claude-4.0-sonnet-200k"

    # DeepSeek Chat models
    DEEPSEEK_CHAT_8K = "deepseek-chat-8k"
    DEEPSEEK_CHAT_16K = "deepseek-chat-16k"
    DEEPSEEK_CHAT_32K = "deepseek-chat-32k"
    DEEPSEEK_CHAT_64K = "deepseek-chat-64k"

    # DeepSeek Reasoner models
    DEEPSEEK_REASONER_4K = "deepseek-reasoner-4k"
    DEEPSEEK_REASONER_8K = "deepseek-reasoner-8k"
    DEEPSEEK_REASONER_16K = "deepseek-reasoner-16k"
    DEEPSEEK_REASONER_32K = "deepseek-reasoner-32k"
    DEEPSEEK_REASONER_64K = "deepseek-reasoner-64k"

    # Gemini 1.5 Flash models
    GEMINI_1_5_FLASH_64K = "gemini-1.5-flash-64k"
    GEMINI_1_5_FLASH_128K = "gemini-1.5-flash-128k"

    # Gemini 1.5 Pro models
    GEMINI_1_5_PRO_2K = "gemini-1.5-pro-2k"
    GEMINI_1_5_PRO_4K = "gemini-1.5-pro-4k"
    GEMINI_1_5_PRO_8K = "gemini-1.5-pro-8k"
    GEMINI_1_5_PRO_16K = "gemini-1.5-pro-16k"
    GEMINI_1_5_PRO_32K = "gemini-1.5-pro-32k"
    GEMINI_1_5_PRO_64K = "gemini-1.5-pro-64k"
    GEMINI_1_5_PRO_128K = "gemini-1.5-pro-128k"

    # Gemini 2.0 Flash models
    GEMINI_2_0_FLASH_128K = "gemini-2.0-flash-128k"

    # Gemini 2.5 Flash Preview models
    GEMINI_2_5_FLASH_PREVIEW_16K = "gemini-2.5-flash-preview-16k"
    GEMINI_2_5_FLASH_PREVIEW_32K = "gemini-2.5-flash-preview-32k"
    GEMINI_2_5_FLASH_PREVIEW_64K = "gemini-2.5-flash-preview-64k"
    GEMINI_2_5_FLASH_PREVIEW_128K = "gemini-2.5-flash-preview-128k"
    GEMINI_2_5_FLASH_PREVIEW_256K = "gemini-2.5-flash-preview-256k"
    GEMINI_2_5_FLASH_PREVIEW_500K = "gemini-2.5-flash-preview-500k"
    GEMINI_2_5_FLASH_PREVIEW_1000K = "gemini-2.5-flash-preview-1000k"

    # Gemini 2.5 Flash Preview Thinking models
    GEMINI_2_5_FLASH_PREVIEW_THINKING_4K = "gemini-2.5-flash-preview-thinking-4k"
    GEMINI_2_5_FLASH_PREVIEW_THINKING_8K = "gemini-2.5-flash-preview-thinking-8k"
    GEMINI_2_5_FLASH_PREVIEW_THINKING_16K = "gemini-2.5-flash-preview-thinking-16k"
    GEMINI_2_5_FLASH_PREVIEW_THINKING_32K = "gemini-2.5-flash-preview-thinking-32k"
    GEMINI_2_5_FLASH_PREVIEW_THINKING_64K = "gemini-2.5-flash-preview-thinking-64k"
    GEMINI_2_5_FLASH_PREVIEW_THINKING_128K = "gemini-2.5-flash-preview-thinking-128k"
    GEMINI_2_5_FLASH_PREVIEW_THINKING_256K = "gemini-2.5-flash-preview-thinking-256k"
    GEMINI_2_5_FLASH_PREVIEW_THINKING_500K = "gemini-2.5-flash-preview-thinking-500k"
    GEMINI_2_5_FLASH_PREVIEW_THINKING_1000K = "gemini-2.5-flash-preview-thinking-1000k"

    # Gemini 2.5 Pro Preview models
    GEMINI_2_5_PRO_PREVIEW_2K = "gemini-2.5-pro-preview-2k"
    GEMINI_2_5_PRO_PREVIEW_4K = "gemini-2.5-pro-preview-4k"
    GEMINI_2_5_PRO_PREVIEW_8K = "gemini-2.5-pro-preview-8k"
    GEMINI_2_5_PRO_PREVIEW_16K = "gemini-2.5-pro-preview-16k"
    GEMINI_2_5_PRO_PREVIEW_32K = "gemini-2.5-pro-preview-32k"
    GEMINI_2_5_PRO_PREVIEW_64K = "gemini-2.5-pro-preview-64k"
    GEMINI_2_5_PRO_PREVIEW_128K = "gemini-2.5-pro-preview-128k"
    GEMINI_2_5_PRO_PREVIEW_256K = "gemini-2.5-pro-preview-256k"
    GEMINI_2_5_PRO_PREVIEW_500K = "gemini-2.5-pro-preview-500k"
    GEMINI_2_5_PRO_PREVIEW_1000K = "gemini-2.5-pro-preview-1000k"

    # GPT 3.5 Turbo models
    GPT_3_5_TURBO_4K = "gpt-3.5-turbo-4k"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"

    # GPT 4 0125 Preview models
    GPT_4_0125_PREVIEW_1K = "gpt-4-0125-preview-1k"
    GPT_4_0125_PREVIEW_2K = "gpt-4-0125-preview-2k"
    GPT_4_0125_PREVIEW_4K = "gpt-4-0125-preview-4k"
    GPT_4_0125_PREVIEW_8K = "gpt-4-0125-preview-8k"
    GPT_4_0125_PREVIEW_16K = "gpt-4-0125-preview-16k"
    GPT_4_0125_PREVIEW_32K = "gpt-4-0125-preview-32k"
    GPT_4_0125_PREVIEW_64K = "gpt-4-0125-preview-64k"
    GPT_4_0125_PREVIEW_128K = "gpt-4-0125-preview-128k"

    # GPT 4.1 models
    GPT_4_1_2K = "gpt-4.1-2k"
    GPT_4_1_4K = "gpt-4.1-4k"
    GPT_4_1_8K = "gpt-4.1-8k"
    GPT_4_1_16K = "gpt-4.1-16k"
    GPT_4_1_32K = "gpt-4.1-32k"
    GPT_4_1_64K = "gpt-4.1-64k"
    GPT_4_1_128K = "gpt-4.1-128k"
    GPT_4_1_256K = "gpt-4.1-256k"
    GPT_4_1_500K = "gpt-4.1-500k"
    GPT_4_1_1000K = "gpt-4.1-1000k"

    # GPT 4.1 Mini models
    GPT_4_1_MINI_4K = "gpt-4.1-mini-4k"
    GPT_4_1_MINI_8K = "gpt-4.1-mini-8k"
    GPT_4_1_MINI_16K = "gpt-4.1-mini-16k"
    GPT_4_1_MINI_32K = "gpt-4.1-mini-32k"
    GPT_4_1_MINI_64K = "gpt-4.1-mini-64k"
    GPT_4_1_MINI_128K = "gpt-4.1-mini-128k"
    GPT_4_1_MINI_256K = "gpt-4.1-mini-256k"
    GPT_4_1_MINI_500K = "gpt-4.1-mini-500k"
    GPT_4_1_MINI_1000K = "gpt-4.1-mini-1000k"

    # GPT 4.1 Nano models
    GPT_4_1_NANO_16K = "gpt-4.1-nano-16k"
    GPT_4_1_NANO_32K = "gpt-4.1-nano-32k"
    GPT_4_1_NANO_64K = "gpt-4.1-nano-64k"
    GPT_4_1_NANO_128K = "gpt-4.1-nano-128k"
    GPT_4_1_NANO_256K = "gpt-4.1-nano-256k"
    GPT_4_1_NANO_500K = "gpt-4.1-nano-500k"
    GPT_4_1_NANO_1000K = "gpt-4.1-nano-1000k"

    # GPT 4o models
    GPT_4O_1K = "gpt-4o-1k"
    GPT_4O_2K = "gpt-4o-2k"
    GPT_4O_4K = "gpt-4o-4k"
    GPT_4O_8K = "gpt-4o-8k"
    GPT_4O_16K = "gpt-4o-16k"
    GPT_4O_32K = "gpt-4o-32k"
    GPT_4O_64K = "gpt-4o-64k"
    GPT_4O_128K = "gpt-4o-128k"

    # GPT 4o Mini models
    GPT_4O_MINI_4K = "gpt-4o-mini-4k"
    GPT_4O_MINI_16K = "gpt-4o-mini-16k"
    GPT_4O_MINI_32K = "gpt-4o-mini-32k"
    GPT_4O_MINI_64K = "gpt-4o-mini-64k"
    GPT_4O_MINI_128K = "gpt-4o-mini-128k"

    # GPT 5 models
    GPT_5_2K = "gpt-5-2k"
    GPT_5_4K = "gpt-5-4k"
    GPT_5_8K = "gpt-5-8k"
    GPT_5_16K = "gpt-5-16k"
    GPT_5_32K = "gpt-5-32k"
    GPT_5_64K = "gpt-5-64k"
    GPT_5_128K = "gpt-5-128k"
    GPT_5_256K = "gpt-5-256k"
    GPT_5_400K = "gpt-5-400k"

    # GPT 5 Mini models
    GPT_5_MINI_4K = "gpt-5-mini-4k"
    GPT_5_MINI_8K = "gpt-5-mini-8k"
    GPT_5_MINI_16K = "gpt-5-mini-16k"
    GPT_5_MINI_32K = "gpt-5-mini-32k"
    GPT_5_MINI_64K = "gpt-5-mini-64k"
    GPT_5_MINI_128K = "gpt-5-mini-128k"
    GPT_5_MINI_256K = "gpt-5-mini-256k"
    GPT_5_MINI_400K = "gpt-5-mini-400k"

    # GPT 5 Nano models
    GPT_5_NANO_16K = "gpt-5-nano-16k"
    GPT_5_NANO_32K = "gpt-5-nano-32k"
    GPT_5_NANO_64K = "gpt-5-nano-64k"
    GPT_5_NANO_128K = "gpt-5-nano-128k"
    GPT_5_NANO_256K = "gpt-5-nano-256k"
    GPT_5_NANO_400K = "gpt-5-nano-400k"

    # O1 models
    O1_1K = "o1-1k"
    O1_2K = "o1-2k"
    O1_4K = "o1-4k"
    O1_8K = "o1-8k"
    O1_16K = "o1-16k"
    O1_32K = "o1-32k"
    O1_64K = "o1-64k"
    O1_128K = "o1-128k"
    O1_200K = "o1-200k"

    # O3 models
    O3_2K = "o3-2k"
    O3_4K = "o3-4k"
    O3_8K = "o3-8k"
    O3_16K = "o3-16k"
    O3_32K = "o3-32k"
    O3_64K = "o3-64k"
    O3_128K = "o3-128k"
    O3_200K = "o3-200k"

    # O3 Mini models
    O3_MINI_2K = "o3-mini-2k"
    O3_MINI_4K = "o3-mini-4k"
    O3_MINI_8K = "o3-mini-8k"
    O3_MINI_16K = "o3-mini-16k"
    O3_MINI_32K = "o3-mini-32k"
    O3_MINI_64K = "o3-mini-64k"
    O3_MINI_128K = "o3-mini-128k"
    O3_MINI_200K = "o3-mini-200k"

    # O3 Pro models
    O3_PRO_2K = "o3-pro-2k"
    O3_PRO_4K = "o3-pro-4k"
    O3_PRO_8K = "o3-pro-8k"
    O3_PRO_16K = "o3-pro-16k"
    O3_PRO_32K = "o3-pro-32k"
    O3_PRO_64K = "o3-pro-64k"
    O3_PRO_128K = "o3-pro-128k"
    O3_PRO_200K = "o3-pro-200k"

    # O4 Mini models
    O4_MINI_2K = "o4-mini-2k"
    O4_MINI_4K = "o4-mini-4k"
    O4_MINI_8K = "o4-mini-8k"
    O4_MINI_16K = "o4-mini-16k"
    O4_MINI_32K = "o4-mini-32k"
    O4_MINI_64K = "o4-mini-64k"
    O4_MINI_128K = "o4-mini-128k"
    O4_MINI_200K = "o4-mini-200k"


def is_valid_model(model: Union[str, ModelType]) -> bool:
    """
    Check if a model string or ModelType enum value is valid.

    Args:
        model: Either a string model name or ModelType enum value

    Returns:
        True if the model is valid, False otherwise
    """
    if isinstance(model, ModelType):
        return True
    return model in MODEL_COSTS


def get_model_cost(model: Union[str, ModelType]) -> int:
    """
    Get the cost for a specific model.

    Args:
        model: Either a string model name or ModelType enum value

    Returns:
        The cost value for the model

    Raises:
        ValueError: If the model is not valid
    """
    if isinstance(model, ModelType):
        model_str = model.value
    else:
        model_str = model

    if model_str not in MODEL_COSTS:
        raise ValueError(f"Invalid model: {model_str}")

    return MODEL_COSTS[model_str]


MODEL_COSTS = {
    # claude-3-haiku
    "claude-3-haiku-4k": 1,
    "claude-3-haiku-8k": 2,  # edited (calc: 1)
    "claude-3-haiku-16k": 3,
    "claude-3-haiku-32k": 4,
    "claude-3-haiku-64k": 6,
    "claude-3-haiku-128k": 12,  # edited (calc: 11)
    # claude-3-opus
    "claude-3-opus-2k": 16,  # edited (calc: 21)
    "claude-3-opus-4k": 40,  # edited (calc: 41)
    "claude-3-opus-8k": 80,  # edited (calc: 82)
    "claude-3-opus-16k": 135,  # edited (calc: 152)
    "claude-3-opus-32k": 225,  # edited (calc: 227)
    "claude-3-opus-64k": 375,  # edited (calc: 377)
    "claude-3-opus-128k": 675,  # edited (calc: 677)
    # claude-3.5-haiku
    "claude-3.5-haiku-2k": 1,
    "claude-3.5-haiku-4k": 2,
    "claude-3.5-haiku-8k": 4,
    "claude-3.5-haiku-16k": 7,  # edited (calc: 9)
    "claude-3.5-haiku-32k": 12,  # edited (calc: 16)
    "claude-3.5-haiku-64k": 20,  # edited (calc: 24)
    "claude-3.5-haiku-128k": 36,  # edited (calc: 40)
    # claude-3.5-sonnet
    "claude-3.5-sonnet-2k": 4,
    "claude-3.5-sonnet-4k": 8,
    "claude-3.5-sonnet-8k": 16,
    "claude-3.5-sonnet-16k": 27,  # edited (calc: 33)
    "claude-3.5-sonnet-32k": 45,  # edited (calc: 61)
    "claude-3.5-sonnet-64k": 75,  # edited (calc: 91)
    "claude-3.5-sonnet-128k": 135,  # edited (calc: 151)
    # claude-3.7-sonnet
    "claude-3.7-sonnet-2k": 4,
    "claude-3.7-sonnet-4k": 8,
    "claude-3.7-sonnet-8k": 16,
    "claude-3.7-sonnet-16k": 27,  # edited (calc: 33)
    "claude-3.7-sonnet-32k": 45,  # edited (calc: 66)
    "claude-3.7-sonnet-64k": 75,  # edited (calc: 132)
    "claude-3.7-sonnet-128k": 135,  # edited (calc: 264)
    "claude-3.7-sonnet-200k": 412,
    # claude-4.0-opus
    "claude-4.0-opus-2k": 21,
    "claude-4.0-opus-4k": 41,
    "claude-4.0-opus-8k": 82,
    "claude-4.0-opus-16k": 165,
    "claude-4.0-opus-32k": 330,
    "claude-4.0-opus-64k": 660,
    "claude-4.0-opus-128k": 1200,
    "claude-4.0-opus-200k": 1538,
    # claude-4.0-sonnet
    "claude-4.0-sonnet-2k": 4,
    "claude-4.0-sonnet-4k": 8,
    "claude-4.0-sonnet-8k": 16,
    "claude-4.0-sonnet-16k": 33,
    "claude-4.0-sonnet-32k": 66,
    "claude-4.0-sonnet-64k": 132,
    "claude-4.0-sonnet-128k": 264,
    "claude-4.0-sonnet-200k": 412,
    # deepseek-chat
    "deepseek-chat-8k": 1,
    "deepseek-chat-16k": 2,  # edited (calc: 1)
    "deepseek-chat-32k": 4,  # edited (calc: 2)
    "deepseek-chat-64k": 8,  # edited (calc: 3)
    # deepseek-reasoner
    "deepseek-reasoner-4k": 1,
    "deepseek-reasoner-8k": 2,  # edited (calc: 1)
    "deepseek-reasoner-16k": 4,  # edited (calc: 1)
    "deepseek-reasoner-32k": 8,  # edited (calc: 2)
    "deepseek-reasoner-64k": 16,  # edited (calc: 3)
    # gemini-1.5-flash
    "gemini-1.5-flash-64k": 1,  # edited (calc: 4)
    "gemini-1.5-flash-128k": 1,  # edited (calc: 7)
    # gemini-1.5-pro
    "gemini-1.5-pro-2k": 3,  # edited (calc: 1)
    "gemini-1.5-pro-4k": 7,  # edited (calc: 3)
    "gemini-1.5-pro-8k": 14,  # edited (calc: 6)
    "gemini-1.5-pro-16k": 24,  # edited (calc: 12)
    "gemini-1.5-pro-32k": 45,  # edited (calc: 22)
    "gemini-1.5-pro-64k": 80,  # edited (calc: 35)
    "gemini-1.5-pro-128k": 150,  # edited (calc: 60)
    # gemini-2.0-flash
    "gemini-2.0-flash-128k": 1,  # edited (calc: 7)
    # gemini-2.5-flash-preview
    "gemini-2.5-flash-preview-16k": 1,
    "gemini-2.5-flash-preview-32k": 3,
    "gemini-2.5-flash-preview-64k": 4,
    "gemini-2.5-flash-preview-128k": 7,
    "gemini-2.5-flash-preview-256k": 13,
    "gemini-2.5-flash-preview-500k": 25,
    "gemini-2.5-flash-preview-1000k": 48,
    # gemini-2.5-flash-preview-thinking
    "gemini-2.5-flash-preview-thinking-4k": 1,
    "gemini-2.5-flash-preview-thinking-8k": 3,
    "gemini-2.5-flash-preview-thinking-16k": 6,
    "gemini-2.5-flash-preview-thinking-32k": 12,
    "gemini-2.5-flash-preview-thinking-64k": 23,
    "gemini-2.5-flash-preview-thinking-128k": 46,
    "gemini-2.5-flash-preview-thinking-256k": 81,
    "gemini-2.5-flash-preview-thinking-500k": 92,
    "gemini-2.5-flash-preview-thinking-1000k": 115,
    # gemini-2.5-pro-preview
    "gemini-2.5-pro-preview-2k": 2,
    "gemini-2.5-pro-preview-4k": 5,
    "gemini-2.5-pro-preview-8k": 10,
    "gemini-2.5-pro-preview-16k": 19,
    "gemini-2.5-pro-preview-32k": 39,
    "gemini-2.5-pro-preview-64k": 78,
    "gemini-2.5-pro-preview-128k": 155,
    "gemini-2.5-pro-preview-256k": 456,
    "gemini-2.5-pro-preview-500k": 647,
    "gemini-2.5-pro-preview-1000k": 1037,
    # gpt-3.5-turbo
    "gpt-3.5-turbo-4k": 1,
    "gpt-3.5-turbo-16k": 4,
    # gpt-4-0125-preview
    "gpt-4-0125-preview-1k": 5,
    "gpt-4-0125-preview-2k": 10,
    "gpt-4-0125-preview-4k": 20,
    "gpt-4-0125-preview-8k": 35,  # edited (calc: 40)
    "gpt-4-0125-preview-16k": 60,  # edited (calc: 76)
    "gpt-4-0125-preview-32k": 120,  # edited (calc: 126)
    "gpt-4-0125-preview-64k": 220,  # edited (calc: 226)
    "gpt-4-0125-preview-128k": 426,
    # gpt-4.1
    "gpt-4.1-2k": 2,
    "gpt-4.1-4k": 5,
    "gpt-4.1-8k": 9,
    "gpt-4.1-16k": 19,
    "gpt-4.1-32k": 38,
    "gpt-4.1-64k": 76,
    "gpt-4.1-128k": 141,
    "gpt-4.1-256k": 221,
    "gpt-4.1-500k": 374,
    "gpt-4.1-1000k": 686,
    # gpt-4.1-mini
    "gpt-4.1-mini-4k": 1,
    "gpt-4.1-mini-8k": 2,
    "gpt-4.1-mini-16k": 4,
    "gpt-4.1-mini-32k": 8,
    "gpt-4.1-mini-64k": 15,
    "gpt-4.1-mini-128k": 28,
    "gpt-4.1-mini-256k": 44,
    "gpt-4.1-mini-500k": 75,
    "gpt-4.1-mini-1000k": 137,
    # gpt-4.1-nano
    "gpt-4.1-nano-16k": 1,
    "gpt-4.1-nano-32k": 2,
    "gpt-4.1-nano-64k": 4,
    "gpt-4.1-nano-128k": 7,
    "gpt-4.1-nano-256k": 11,
    "gpt-4.1-nano-500k": 19,
    "gpt-4.1-nano-1000k": 34,
    # gpt-4o
    "gpt-4o-1k": 3,  # edited (calc: 1)
    "gpt-4o-2k": 5,  # edited (calc: 3)
    "gpt-4o-4k": 10,  # edited (calc: 6)
    "gpt-4o-8k": 20,  # edited (calc: 12)
    "gpt-4o-16k": 40,  # edited (calc: 24)
    "gpt-4o-32k": 60,  # edited (calc: 48)
    "gpt-4o-64k": 120,  # edited (calc: 88)
    "gpt-4o-128k": 160,  # edited (calc: 138)
    # gpt-4o-mini
    "gpt-4o-mini-4k": 1,
    "gpt-4o-mini-16k": 2,  # edited (calc: 1)
    "gpt-4o-mini-32k": 6,  # edited (calc: 3)
    "gpt-4o-mini-64k": 10,  # edited (calc: 5)
    "gpt-4o-mini-128k": 15,  # edited (calc: 8)
    # gpt-5
    "gpt-5-2k": 2,
    "gpt-5-4k": 5,
    "gpt-5-8k": 10,
    "gpt-5-16k": 19,
    "gpt-5-32k": 39,
    "gpt-5-64k": 78,
    "gpt-5-128k": 155,
    "gpt-5-256k": 310,
    "gpt-5-400k": 484,
    # gpt-5-mini
    "gpt-5-mini-4k": 1,
    "gpt-5-mini-8k": 2,
    "gpt-5-mini-16k": 4,
    "gpt-5-mini-32k": 8,
    "gpt-5-mini-64k": 15,
    "gpt-5-mini-128k": 31,
    "gpt-5-mini-256k": 62,
    "gpt-5-mini-400k": 97,
    # gpt-5-nano
    "gpt-5-nano-16k": 1,
    "gpt-5-nano-32k": 2,
    "gpt-5-nano-64k": 3,
    "gpt-5-nano-128k": 6,
    "gpt-5-nano-256k": 12,
    "gpt-5-nano-400k": 19,
    # o1
    "o1-1k": 8,  # edited (calc: 9)
    "o1-2k": 16,  # edited (calc: 18)
    "o1-4k": 36,
    "o1-8k": 72,  # edited (calc: 71)
    "o1-16k": 120,  # edited (calc: 143)
    "o1-32k": 200,  # edited (calc: 285)
    "o1-64k": 360,  # edited (calc: 570)
    "o1-128k": 600,  # edited (calc: 1140)
    "o1-200k": 1781,
    # o3
    "o3-2k": 2,
    "o3-4k": 5,
    "o3-8k": 9,
    "o3-16k": 19,
    "o3-32k": 38,
    "o3-64k": 76,
    "o3-128k": 152,
    "o3-200k": 238,
    # o3-mini
    "o3-mini-2k": 1,
    "o3-mini-4k": 2,  # edited (calc: 3)
    "o3-mini-8k": 4,  # edited (calc: 5)
    "o3-mini-16k": 8,  # edited (calc: 10)
    "o3-mini-32k": 16,  # edited (calc: 21)
    "o3-mini-64k": 24,  # edited (calc: 42)
    "o3-mini-128k": 48,  # edited (calc: 84)
    "o3-mini-200k": 131,
    # o3-pro
    "o3-pro-2k": 24,
    "o3-pro-4k": 48,
    "o3-pro-8k": 95,
    "o3-pro-16k": 190,
    "o3-pro-32k": 380,
    "o3-pro-64k": 760,
    "o3-pro-128k": 1520,
    "o3-pro-200k": 2375,
    # o4-mini
    "o4-mini-2k": 1,
    "o4-mini-4k": 3,
    "o4-mini-8k": 5,
    "o4-mini-16k": 10,
    "o4-mini-32k": 21,
    "o4-mini-64k": 42,
    "o4-mini-128k": 84,
    "o4-mini-200k": 131,
}
