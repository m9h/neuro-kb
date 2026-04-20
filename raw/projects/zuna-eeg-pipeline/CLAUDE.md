# Project: Zuna EEG Imputation Pipeline (ds000117)

## Environment & Hardware Gotchas
- GPU Architecture: This container runs on an NVIDIA DGX Spark with Blackwell (sm_121) GPUs.
- The Compiler Bug: The bundled PyTorch LLVM compiler does not yet support sm_121, causing a SIGKILL 9 during JIT compilation. You MUST ensure os.environ["TORCHDYNAMO_DISABLE"] = "1" is set to bypass this, especially when Zuna spawns subprocesses.
- The PyTorch Bug (VENDORED): The container uses PyTorch 2.6. The Zuna package has a stale in_order=False kwarg in its DataLoader that must be patched out. You MUST patch the vendored version inside the Zuna tree, NOT the global site-packages. The exact file to patch is: ~/.local/lib/python3.12/site-packages/zuna/inference/AY2l/lingua/apps/AY2latent_bci/eeg_data.py
- DataLoader num_workers (OOM Prevention): NEVER use os.cpu_count() or rely on default num_workers when constructing a DataLoader for EEG datasets. Always hardcode num_workers to a maximum of 4. High-core-count hardware like the DGX will spawn excessive worker processes and cause OOM crashes if this is left to defaults.

## Commands
- Run Batch: python zuna_batch_eval.py