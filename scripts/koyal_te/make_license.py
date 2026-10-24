#!/usr/bin/env python3
"""
Generate LICENSE-koyal-te-120m.txt for release packaging.

Required components:
1. CC-BY-4.0 notice
2. Attribution line from the model card
3. Notice: "encoder from AI4Bharat MahaDhwani, MIT"
"""

import argparse
from pathlib import Path

CC_BY_4_NOTICE = """Creative Commons Attribution 4.0 International (CC-BY-4.0)

You are free to:
- Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
- Adapt — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode
"""

MODEL_ATTRIBUTION = """Model: adalat-ai/koyal-te-120m-1.0
Authors: Adalat AI (https://huggingface.co/adalat-ai)
Repository: https://huggingface.co/adalat-ai/koyal-te-120m-1.0
Base architecture / Encoder: encoder from AI4Bharat MahaDhwani, MIT
License: Creative Commons Attribution 4.0 International (CC-BY-4.0)

Attribution line:
Koyal te-120M by Adalat AI (https://huggingface.co/adalat-ai/koyal-te-120m-1.0), licensed under CC-BY-4.0. Encoder from AI4Bharat MahaDhwani, MIT.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=None, help="Optional downloaded README.md from model card")
    parser.add_argument("--out", required=True, type=Path, help="Output LICENSE-koyal-te-120m.txt path")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    text_parts = [
        "================================================================================",
        "ADALAT AI KOYAL TE-120M LICENSE AND ATTRIBUTION NOTICE",
        "================================================================================",
        "",
        MODEL_ATTRIBUTION.strip(),
        "",
        "--------------------------------------------------------------------------------",
        "CC-BY-4.0 NOTICE",
        "--------------------------------------------------------------------------------",
        "",
        CC_BY_4_NOTICE.strip(),
        "",
        "--------------------------------------------------------------------------------",
        "UPSTREAM ENCODER NOTICE",
        "--------------------------------------------------------------------------------",
        "",
        "The acoustic encoder of this model is derived from AI4Bharat MahaDhwani, released under the MIT License.",
        "Copyright (c) 2023-2024 AI4Bharat.",
        "",
    ]

    if args.readme and args.readme.exists():
        readme_content = args.readme.read_text(encoding="utf-8", errors="replace")
        text_parts.extend([
            "--------------------------------------------------------------------------------",
            "MODEL CARD TEXT (from Hugging Face repository)",
            "--------------------------------------------------------------------------------",
            "",
            readme_content.strip(),
            "",
        ])

    args.out.write_text("\n".join(text_parts), encoding="utf-8")
    print(f"Wrote license notice to {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    main()
