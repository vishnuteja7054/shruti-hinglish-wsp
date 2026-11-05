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

    api = HfApi(token=args.token)
    user_info = api.whoami()
    print(f"Authenticated as Hugging Face user: {user_info['name']}")
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
        print("Uploaded README.md (Model Card)", flush=True)

    orig_dir = ROOT / "checkpoints" / "original"
    exp_dir = ROOT / "checkpoints" / "experiments"

    # 1. Upload flagship Phase 11 E2E FP16 .cact at root for 1-click download
    best_cact = exp_dir / "shruti_wsp_phase11_e16_e2e_fp16.cact"
    if best_cact.exists():
        print(f"Uploading root flagship checkpoint {best_cact.name} ({best_cact.stat().st_size / (1024*1024):.1f} MB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(best_cact),
            path_in_repo="shruti_wsp_phase11_e16_e2e_fp16.cact",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload flagship Phase 11 E2E FP16 .cact checkpoint at root",
        )

    # 2. Upload all .cact binary checkpoints + SHA256SUMS first (29.5 MB each)
    for folder_name, d in [("original", orig_dir), ("experiments", exp_dir)]:
        for f in sorted(d.iterdir()):
            if f.suffix in (".cact", ".txt"):
                print(f"Uploading {folder_name}/{f.name} ({f.stat().st_size / (1024*1024):.1f} MB)...", flush=True)
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f"{folder_name}/{f.name}",
                    repo_id=args.repo_id,
                    repo_type="model",
                    commit_message=f"Upload {folder_name}/{f.name}",
                )

    # 3. Upload .safetensors checkpoints unless --cact-only is set
    if not args.cact_only:
        for folder_name, d in [("experiments", exp_dir), ("original", orig_dir)]:
            for f in sorted(d.iterdir()):
                if f.suffix == ".safetensors":
                    print(f"Uploading {folder_name}/{f.name} ({f.stat().st_size / (1024*1024):.1f} MB)...", flush=True)
                    api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=f"{folder_name}/{f.name}",
                        repo_id=args.repo_id,
                        repo_type="model",
                        commit_message=f"Upload {folder_name}/{f.name}",
                    )

    print(f"\nAll checkpoints successfully uploaded to https://huggingface.co/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
