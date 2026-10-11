"""
LoRA Catalog - an index of what every LoRA in a local library is and how it is used.

Phase 1 (this code): the INDEXER. It walks the library and gathers the evidence each file already has, from the
cheapest sources first, without touching the network:

    LoRA Manager's per-file .metadata.json   sha256 (saves hashing 1.4 TB), favourites, notes, Civitai fields if filled
    LoRA Manager's offline Civitai database  description, tags, trained words, base model, licence, by sha256
    the Civitai downloader's sidecars        version id from the folder layout; gallery prompts with the weights used
    the safetensors header                   kohya ss_tag_frequency, modelspec, network dim, a tensor-key signature
    ai-toolkit training folders              trigger word, architecture and the training captions, for your own LoRAs

Later phases (spec: video_prompter/dev/LORA_CATALOG_SPEC.md): a resumable Civitai crawl for what is still
unidentified, the refiner that writes each LoRA's card (kind, scope, effect, overrides), a library browser, and
LoRA suggestion in the Wildcard Composer.

LoRA Manager is GPL-3. This package reads its DATA files only and contains none of its code.

(c) Eric Hiss (GitHub: EricRollei). Dual license, see LICENSE.
"""

__version__ = "0.1.0"
