"""Simple web UI for PageIndex.

Provides a single page where a user can upload a PDF or Markdown file,
process it with the existing PageIndex pipeline, and download the resulting
JSON tree structure.

Run with:
    python -m webapp.app
or
    flask --app webapp.app run
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from pageindex import page_index_main
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader

# Limit upload size (default 50 MB; override via PAGEINDEX_MAX_UPLOAD_MB)
_DEFAULT_MAX_UPLOAD_MB = 50
_MAX_UPLOAD_MB = int(os.environ.get("PAGEINDEX_MAX_UPLOAD_MB", _DEFAULT_MAX_UPLOAD_MB))

ALLOWED_PDF_EXTS = {".pdf"}
ALLOWED_MD_EXTS = {".md", ".markdown"}
ALLOWED_EXTS = ALLOWED_PDF_EXTS | ALLOWED_MD_EXTS


def _ext(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def _safe_stem(filename: str) -> str:
    """Return a filesystem-safe stem derived from the uploaded filename."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    safe = "".join(c for c in stem if c.isalnum() or c in ("-", "_")).strip("-_")
    return safe or "document"


def _truthy_or_none(value):
    """Return 'yes'/'no' for checkbox-like inputs, or None to keep config default."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    v = str(value).strip().lower()
    if v in ("yes", "true", "on", "1"):
        return "yes"
    if v in ("no", "false", "off", "0"):
        return "no"
    return None


def _process_pdf(pdf_path: str, form) -> dict:
    user_opt = {
        "model": form.get("model") or None,
        "if_add_node_id": _truthy_or_none(form.get("if_add_node_id")),
        "if_add_node_summary": _truthy_or_none(form.get("if_add_node_summary")),
        "if_add_doc_description": _truthy_or_none(form.get("if_add_doc_description")),
        "if_add_node_text": _truthy_or_none(form.get("if_add_node_text")),
    }
    opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})
    return page_index_main(pdf_path, opt)


def _process_md(md_path: str, form) -> dict:
    config_loader = ConfigLoader()
    user_opt = {
        "model": form.get("model") or None,
        "if_add_node_id": _truthy_or_none(form.get("if_add_node_id")),
        "if_add_node_summary": _truthy_or_none(form.get("if_add_node_summary")),
        "if_add_doc_description": _truthy_or_none(form.get("if_add_doc_description")),
        "if_add_node_text": _truthy_or_none(form.get("if_add_node_text")),
    }
    opt = config_loader.load({k: v for k, v in user_opt.items() if v is not None})

    return asyncio.run(
        md_to_tree(
            md_path=md_path,
            if_add_node_summary=opt.if_add_node_summary,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id,
            model=opt.model,
        )
    )


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_MB * 1024 * 1024

    @app.get("/")
    def index():
        return render_template("index.html", max_upload_mb=_MAX_UPLOAD_MB)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/process")
    def process():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "No file uploaded. Please choose a PDF or Markdown file."}), 400

        ext = _ext(uploaded.filename)
        if ext not in ALLOWED_EXTS:
            return jsonify({
                "error": f"Unsupported file type '{ext}'. Allowed: .pdf, .md, .markdown"
            }), 400

        stem = _safe_stem(uploaded.filename)

        # Save upload to a unique temp directory; clean up after processing.
        with tempfile.TemporaryDirectory(prefix="pageindex_upload_") as tmpdir:
            saved_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{ext}")
            uploaded.save(saved_path)

            try:
                if ext in ALLOWED_PDF_EXTS:
                    tree = _process_pdf(saved_path, request.form)
                else:
                    tree = _process_md(saved_path, request.form)
            except Exception as exc:  # noqa: BLE001 - surface error message to client
                app.logger.exception("Processing failed")
                return jsonify({"error": f"Processing failed: {exc}"}), 500

        # Serialize result and stream as a download.
        payload = json.dumps(tree, indent=2, ensure_ascii=False).encode("utf-8")
        buffer = io.BytesIO(payload)
        download_name = f"{stem}_structure.json"
        return send_file(
            buffer,
            mimetype="application/json",
            as_attachment=True,
            download_name=download_name,
        )

    @app.errorhandler(413)
    def _too_large(_e):
        return jsonify({
            "error": f"File is too large. Limit is {_MAX_UPLOAD_MB} MB."
        }), 413

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("PAGEINDEX_HOST", "127.0.0.1")
    port = int(os.environ.get("PAGEINDEX_PORT", "5000"))
    debug = os.environ.get("PAGEINDEX_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
