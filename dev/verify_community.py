"""Offline checks for community.py: prompt cleaning, prompt style, weights from <lora:> tags, percentiles."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lora_catalog import community as c        # noqa: E402

FAILS = []


def check(label, ok, extra=""):
    print(("  ok   " if ok else "  FAIL ") + label + ("" if ok else f"   {extra!r}"))
    if not ok:
        FAILS.append(label)


cleaned = c.clean_prompt("<lora:comic_v2:0.8> a (hero:1.3) on a ((rooftop)), [rain], embedding:badhands BREAK night <hypernet:x:1>")
check("clean strips tags and weights", cleaned == "a hero on a rooftop, rain, , night".replace(", ,", ",") or
      ("lora" not in cleaned and "1.3" not in cleaned and "embedding" not in cleaned and "hero" in cleaned), cleaned)
check("tag style", c.prompt_style("score_9, score_8_up, 1girl, solo, red hair, smile") == "tags")
check("tag style by shape", c.prompt_style("red hair, blue eyes, smile, beach, sunset, bikini, looking at viewer") == "tags")
check("natural style", c.prompt_style("A weathered sea captain stands on the deck of his ship as a storm rolls in, rain "
                                      "streaking across his face.") == "natural")
item = {"meta": {"prompt": "a cat <lora:3160971_gilgbertzstyle_epoch_9:0.65> sitting"}, "modelVersionIds": []}


class NoIndex:
    by_hash, by_version = {}, {}

    def key_for(self, *a):
        return None

    def owned(self, key):
        return None


check("weight from <lora:> tag", c._weight_for(item, 1, ["3160971_gilgbertzstyle_epoch_9"], NoIndex()) == 0.65)
check("percentiles", c._percentiles([1, 2, 3, None]) == [1 / 6, 0.5, 5 / 6, None])
check("pct median", c._pct([1, 2, 3, 4], 0.5) == 2.5)
check("nsfw words", bool(c.NSFW_WORDS.search("a nude figure")) and not c.NSFW_WORDS.search("a portrait of a sailor"))
check("similar", c._similar("a red fox in snow at dawn", "a red fox in the snow at dawn") and not c._similar("a fox", "a ship at sea"))
print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
