"""
Pipeline Log Handler
====================
Captures [Pipeline] logs in real-time and sends them to progress tracker.
Enables real-time display of actual pipeline step messages.
"""

import logging
import re
from typing import Optional


class PipelineLogHandler(logging.Handler):
    """Custom logging handler that captures [Pipeline] logs and stores as events."""

    def __init__(self, progress_tracker, doc_id: Optional[str] = None):
        """
        Args:
            progress_tracker: IngestionProgressTracker singleton
            doc_id: Document ID being ingested (can be set later)
        """
        super().__init__()
        self.progress_tracker = progress_tracker
        self.doc_id = doc_id
        self.log_buffer = []

    def set_doc_id(self, doc_id: str):
        """Set the document ID after it's generated."""
        self.doc_id = doc_id

    def emit(self, record):
        """Called when a log record is emitted."""
        try:
            message = record.getMessage()

            # Only process [Pipeline] logs
            if "[Pipeline]" not in message:
                return

            if not self.doc_id:
                return

            # Clean up message
            clean_msg = message.replace("[Pipeline]", "").strip()

            # Extract step name
            step_name = self._extract_step_name(clean_msg)

            if step_name:
                status = self._determine_status(clean_msg)

                self.progress_tracker.record_event(
                    self.doc_id,
                    stage=step_name,
                    status=status,
                    message=clean_msg,
                    duration_ms=0,
                )

        except Exception:
            # Silently ignore errors in log handling
            pass

    def _extract_step_name(self, message: str) -> Optional[str]:
        """Extract step name from log message."""

        # Pattern: "Step N: Description"
        step_match = re.search(r"Step \d+\s*[:\(-]?\s*(.+?)(?:\(|\n|$)", message)
        if step_match:
            return f"Step: {step_match.group(1).strip()}"

        # Pattern: "✓ Description"
        if message.startswith("✓"):
            desc = message.replace("✓", "").strip()
            parts = re.split(r"[:\-]", desc)
            if parts:
                return parts[0].strip()

        # Warning patterns
        if "⚠" in message or "Warning" in message:
            return "Warning"

        if "Duplicate" in message:
            return "Duplicate Check"

        if "Generated" in message:
            return "ID Generated"

        if "error" in message.lower() or "failed" in message.lower():
            return "Error"

        # Default extraction
        match = re.match(r"([A-Z][^:\-\.]*)", message)
        if match:
            name = match.group(1).strip()
            if len(name) > 3:
                return name

        return None

    def _determine_status(self, message: str) -> str:
        """Determine status from log message."""
        if "✓" in message or "complete" in message.lower() or "ready" in message.lower():
            return "completed"
        elif "Step" in message:
            return "in_progress"
        elif "error" in message.lower() or "failed" in message.lower():
            return "failed"
        else:
            return "in_progress"


def setup_pipeline_logging(
    progress_tracker, doc_id: Optional[str] = None
) -> PipelineLogHandler:
    """
    Setup pipeline logging to capture real-time progress.

    Args:
        progress_tracker: IngestionProgressTracker singleton
        doc_id: Document ID (optional, can be set later)

    Returns:
        PipelineLogHandler instance
    """
    handler = PipelineLogHandler(progress_tracker, doc_id)
    handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    return handler