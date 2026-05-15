# Agent Mandates for similar-images

This project has strict architectural and workflow requirements. All AI agents MUST adhere to the following rules:

## 1. Package Management & Tooling
- **Manager:** NEVER use `pip`. Use **`uv`** for all dependency and execution tasks.
- **Adding Dependencies:** Use `uv add <package>`.
- **Running Tools:** Use `uv run similar-images ...` or `uv run python ...`.
- **GUI Framework:** Use `customtkinter` for all GUI elements to maintain a modern aesthetic.

## 2. Implementation Integrity (CRITICAL)
- **Surgical Edits:** ONLY modify code directly related to the requested task. Do not perform "cleanup" or refactoring of unrelated files or functions.
- **Preserve User Edits:** The user often fine-tunes thresholds, weights, or logic. **NEVER overwrite a user's specific value (e.g., a default threshold) unless explicitly instructed to do so.**
- **Atomic Changes:** Ensure changes are complete and functional within their scope, but do not bleed into other layers of the application.

## 3. Similarity Logic Standards
- **Multi-Resolution:** Maintain the 32x32 stretched vs. 128x128 structural separation.
- **Consistency:** If adding a new feature, ensure it is integrated into the CLI, the GUI, and the HTML report summary.

## 4. Documentation
- Update `README.md` if CLI options or default behaviors change.
- Ensure the "Similarity Approach" section in `README.md` accurately reflects the latest implementation in `features.py` and `similarity.py`.
