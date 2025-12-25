import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from src.voice_synthesis import (
    VoiceManager,
    VoiceProviderConfig,
    VoiceProviderType,
    VoiceRequest
)

async def verify_voice_synthesis():
    """Verify voice synthesis with configured providers."""
    print("Verifying Voice Synthesis...")

    # Configure voice providers
    voice_configs = []

    # ElevenLabs - Check env var manually if settings not loaded right in this context
    eleven_key = settings.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
    if eleven_key:
        print("Configuring ElevenLabs...")
        voice_configs.append(VoiceProviderConfig(
            provider_type=VoiceProviderType.ELEVENLABS,
            api_key=eleven_key
        ))
    else:
        print("Skipping ElevenLabs (no API Key)")

    # Fish Audio
    fish_key = settings.fish_audio_api_key or os.environ.get("FISH_AUDIO_API_KEY")
    if fish_key:
        print("Configuring Fish Audio...")
        voice_configs.append(VoiceProviderConfig(
            provider_type=VoiceProviderType.FISH_AUDIO,
            api_key=fish_key
        ))
    else:
        print("Skipping Fish Audio (no API Key)")

    # Ollama TTS
    print(f"Configuring Ollama TTS at {settings.ollama_tts_url}...")
    voice_configs.append(VoiceProviderConfig(
        provider_type=VoiceProviderType.OLLAMA_TTS,
        base_url=settings.ollama_tts_url
    ))

    # Initialize Manager
    manager = VoiceManager(
        provider_configs=voice_configs,
        cache_dir=Path("./test_output/audio_cache"),
        prefer_local=True
    )

    await manager.initialize()

    # Test Synthesis
    test_text = "Welcome to the adventure. This is a voice test."
    print(f"\nSynthesizing: '{test_text}'")

    # Get status
    status = await manager.get_system_status()
    print("\nSystem Status:")
    for provider, p_status in status["providers"].items():
        print(f"- {provider}: {'Available' if p_status['available'] else 'Unavailable'}")

    # Test specific providers if available
    for provider_type in [VoiceProviderType.OLLAMA_TTS, VoiceProviderType.ELEVENLABS, VoiceProviderType.FISH_AUDIO]:
        provider = manager.get_provider(provider_type)
        if provider and provider.is_available:
            print(f"\nTesting {provider_type.value}...")
            try:
                # Force specific provider in profile or request?
                # VoiceManager selects best match, but we can't force provider easily in public API without specific profile
                # We can use internal method for verification or create a profile for it.

                # Create temp profile for this provider
                profile = await manager.create_profile(
                    name=f"test_{provider_type.value}",
                    provider_voice_ids={provider_type: "default"} # Placeholder
                )

                # Hack: Update the profile to force this provider if possible?
                # Actually, VoiceManager.synthesize uses _get_available_provider which uses priority.
                # To test specific provider, we might need to call provider directly or use a profile that ONLY has this provider voice mapped?
                # The _get_available_provider doesn't filter by profile unless we enforce it.

                # Let's call provider directly for verification script
                request = VoiceRequest(text=test_text)
                response = await provider.synthesize(request, profile)

                if response.success:
                    print(f"✅ Success! Audio size: {len(response.audio_data)} bytes")
                    output_file = Path(f"./test_output/test_{provider_type.value}.{getattr(response, 'format', 'mp3')}")
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    output_file.write_bytes(response.audio_data)
                    print(f"Saved to {output_file}")
                else:
                    print(f"❌ Failed: {response.error}")

            except Exception as e:
                print(f"❌ Exception: {e}")

    await manager.shutdown()
    print("\nVerification Complete.")

if __name__ == "__main__":
    asyncio.run(verify_voice_synthesis())
