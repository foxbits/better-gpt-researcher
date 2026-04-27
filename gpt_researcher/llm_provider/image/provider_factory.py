"""
Image provider factory for GPT Researcher.

This module provides a factory function to select the appropriate image provider
based on the IMAGE_GENERATION_PROVIDER environment variable.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_image_provider(
        provider_type: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional[object]:
    """Get the appropriate image provider based on configuration.

    Returns:
        An image provider instance (ImageGeneratorProvider or OpenAIImageProvider),
        or None if image generation is disabled.

    The provider is selected based on the provider_type setting:
    - "google" or not set: Uses Google's ImageGeneratorProvider (default)
    - "openai": Uses OpenAI-compatible OpenAIImageProvider
    """

    if provider_type == "openai":
        try:
            from .openai_image_provider import OpenAIImageProvider
            return OpenAIImageProvider(model_name=model_name, base_url=base_url, api_key=api_key)
        except ImportError as e:
            logger.error(f"Failed to import OpenAIImageProvider: {e}")
            return None

    if provider_type == "google":
        try:
            from .image_generator import ImageGeneratorProvider
            return ImageGeneratorProvider(model_name=model_name, api_key=api_key)
        except ImportError as e:
            logger.error(f"Failed to import ImageGeneratorProvider: {e}")
            return None

    logger.warning(f"Unknown IMAGE_GENERATION_PROVIDER: {provider_type}, using google")
    try:
        from .image_generator import ImageGeneratorProvider
        return ImageGeneratorProvider
    except ImportError:
        return None
