"""
OpenAI-compatible image generation provider for GPT Researcher.

Supports any OpenAI-compatible API endpoint (nano-gpt.com, local servers, etc.)
for image generation via the /v1/images/generations endpoint.
"""

import asyncio
import base64
import hashlib
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OpenAIImageProvider:
    """Provider for generating images using OpenAI-compatible APIs.

    Supports any provider that implements the OpenAI images/generations endpoint:
    - OpenAI / wavespeed / nano-gpt.com
    - Local image generation servers
    - Other OpenAI-compatible providers

    Attributes:
        model_name: The model to use (e.g., "dall-e-3", "flux", etc.)
        api_key: API key for authentication
        base_url: Base URL for the API endpoint
        output_dir: Directory to save generated images
    """

    DEFAULT_MODEL = "qwen-image-1.0"  # Default to a more universally supported model for compatibility

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        output_dir: str = "outputs",
    ):
        """Initialize the OpenAIImageProvider.

        Args:
            model_name: The model to use. Defaults to "dall-e-3".
            api_key: API key. If not provided, reads from IMAGE_API_KEY or OPENAI_API_KEY.
            base_url: Base URL for the API. If not provided, reads from IMAGE_GENERATION_BASE_URL.
            output_dir: Base directory for outputs.
        """
        self.model_name = model_name or os.getenv("IMAGE_GENERATION_MODEL", self.DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("IMAGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("IMAGE_GENERATION_BASE_URL", "").rstrip("/")
        self.output_dir = Path(output_dir)

        if not self.api_key:
            logger.warning(
                "No API key found for image generation. "
                "Set IMAGE_API_KEY or OPENAI_API_KEY environment variable."
            )
        if not self.base_url:
            logger.warning(
                "No base URL configured for image generation. "
                "Set IMAGE_GENERATION_BASE_URL environment variable."
            )

    def _ensure_client(self):
        """Ensure the OpenAI client is initialized."""
        if not hasattr(self, '_client') or self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                )
                logger.info(f"Initialized OpenAI image generation: {self.base_url or 'official API'}")
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI-compatible image generation. "
                    "Install with: pip install openai"
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise

    def _ensure_output_dir(self, research_id: str = "") -> Path:
        """Ensure the output directory exists."""
        if research_id:
            output_path = self.output_dir / "images" / research_id
        else:
            output_path = self.output_dir / "images"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _generate_image_filename(self, prompt: str, index: int = 0) -> str:
        """Generate unique filename based on prompt hash."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return f"img_{prompt_hash}_{index}.png"

    def _build_prompt(self, prompt: str, context: str = "", style: str = "dark") -> str:
        """Build prompt with styling (simplified for OpenAI compat)."""
        styled = prompt
        if context:
            styled += f"\n\nContext: {context[:200]}"
        return styled

    async def generate_image(
        self,
        prompt: str,
        context: str = "",
        research_id: str = "",
        aspect_ratio: str = "1:1",
        num_images: int = 1,
        style: str = "dark",
    ) -> List[Dict[str, Any]]:
        """Generate images using OpenAI-compatible endpoint.

        Args:
            prompt: The image generation prompt
            context: Additional context
            research_id: Research ID for organizing output
            aspect_ratio: Aspect ratio (e.g., "1024x1024", "1792x1024")
            num_images: Number of images to generate
            style: Image style (passed to provider if supported)

        Returns:
            List of image info dictionaries with absolute paths
        """
        if not self.api_key:
            logger.warning("No API key configured for image generation")
            return []

        if not self.base_url:
            logger.warning("No base URL configured for image generation")
            return []

        self._ensure_client()
        output_path = self._ensure_output_dir(research_id)

        full_prompt = self._build_prompt(prompt, context, style)

        try:
            size_map = {
                "1:1": "1024x1024",
                "16:9": "1792x1024",
                "9:16": "1024x1792",
            }
            size = size_map.get(aspect_ratio, "1024x1024")

            response = await asyncio.to_thread(
                self._client.images.generate,
                model=self.model_name,
                prompt=full_prompt,
                size=size,
                n=min(num_images, 4),
                response_format="b64_json",
            )

            generated_images = []
            for i, image_data in enumerate(response.data):
                filename = self._generate_image_filename(prompt, i)
                filepath = output_path / filename

                image_bytes = base64.b64decode(image_data.b64_json)

                with open(filepath, 'wb') as f:
                    f.write(image_bytes)

                absolute_path = filepath.resolve()
                web_url = f"/outputs/images/{research_id}/{filename}" if research_id else f"/outputs/images/{filename}"

                generated_images.append({
                    "path": str(absolute_path),
                    "url": web_url,
                    "absolute_url": str(absolute_path),
                    "prompt": prompt,
                    "alt_text": self._generate_alt_text(prompt),
                })

                logger.info(f"Generated image saved to: {filepath}")

            return generated_images

        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            return []

    def _generate_alt_text(self, prompt: str) -> str:
        """Generate accessible alt text from the prompt."""
        clean_prompt = prompt.replace('\n', ' ').strip()
        if len(clean_prompt) > 120:
            clean_prompt = clean_prompt[:117] + "..."
        return f"Illustration: {clean_prompt}"

    def is_available(self) -> bool:
        """Check if image generation is available."""
        if not self.api_key or not self.base_url:
            return False
        try:
            self._ensure_client()
            return True
        except Exception:
            return False

    @classmethod
    def from_config(cls, config) -> Optional["OpenAIImageProvider"]:
        """Create from Config object."""
        enabled = getattr(config, 'image_generation_enabled', False)
        provider = getattr(config, 'image_generation_provider', 'openai')

        if not enabled or provider != 'openai':
            return None

        model = getattr(config, 'image_generation_model', None)
        base_url = getattr(config, 'image_generation_base_url', None)

        return cls(
            model_name=model or cls.DEFAULT_MODEL,
            base_url=base_url,
        )