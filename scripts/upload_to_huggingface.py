#!/usr/bin/env python3
"""Upload Shruti WSP Checkpoints (.cact and .safetensors) and Model Card to Hugging Face Hub."""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Upload Shruti WSP checkpoints to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID, e.g. username/shruti-hinglish-wsp")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face write token (or set HF_TOKEN)")
    parser.add_argument("--private", action="store_true", help="Create private Hugging Face repo")
    parser.add_argument("--cact-only", action="store_true", help="Upload only .cact files (skip 212MB .safetensors)")
    args = parser.parse_args()

    if not args.token:
        raise ValueError("Please provide --token or set HF_TOKEN environment variable.")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Verified Hugging Face model repository: https://huggingface.co/{args.repo_id}")

    # Upload HF Model Card as README.md
    card_path = ROOT / "checkpoints" / "HF_MODEL_CARD.md"
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload Shruti-Hinglish-WSP Model Card and SHA-256 inventory",
        )
        print("Uploaded README.md (Model Card)")

    # Upload original base checkpoints
    orig_dir = ROOT / "checkpoints" / "original"
    for f in sorted(orig_dir.iterdir()):
        if args.cact_only and f.suffix == ".safetensors":
            continue
        if f.suffix in (".cact", ".safetensors", ".txt"):
            print(f"Uploading original/{f.name} ({f.stat().st_size / (1024*1024):.1f} MB)...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"original/{f.name}",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload original/{f.name}",
            )

    # Upload all experiment checkpoints
    exp_dir = ROOT / "checkpoints" / "experiments"
    for f in sorted(exp_dir.iterdir()):
        if args.cact_only and f.suffix == ".safetensors":
            continue
        if f.suffix in (".cact", ".safetensors"):
            print(f"Uploading experiments/{f.name} ({f.stat().st_size / (1024*1024):.1f} MB)...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"experiments/{f.name}",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload {f.name}",
            )

    # Also upload the top recommended production .cact at the root of the HF repo for 1-click download
    best_cact = exp_dir / "shruti_wsp_phase11_e16_e2e_fp16.cact"
    if best_cact.exists():
        print(f"Uploading root production checkpoint {best_cact.name}...")
        api.upload_file(
            path_or_fileobj=str(best_cact),
            path_in_repo="shruti_wsp_phase11_e16_e2e_fp16.cact",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload flagship Phase 11 E2E FP16 .cact checkpoint at root",
        )

    print(f"\nAll checkpoints successfully uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
