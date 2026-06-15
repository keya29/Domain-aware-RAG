"""
URL PDF Downloader Module
========================
Handles downloading PDFs from URLs, validation, and temporary file management.
Downloaded PDFs are stored temporarily and passed to data_extraction.py.
Uses URL filename as the PDF identifier.
"""

import os
import logging
import requests
import tempfile
from typing import Tuple, Iterator
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class URLPDFDownloader:
    """Download and validate PDFs from URLs."""

    TEMP_DIR: str = tempfile.gettempdir()

    REQUEST_TIMEOUT: int = 30  # seconds
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB
    CHUNK_SIZE: int = 8192  # bytes

    VALID_MIME_TYPES: list[str] = [
        "application/pdf",
        "application/x-pdf",
        "application/x-bzpdf",
        "application/x-gzpdf",
    ]

    @staticmethod
    def download(url: str) -> Tuple[str, str]:
        """Download PDF from URL and save to temporary location."""
        try:
            URLPDFDownloader._validate_url(url)
            logger.info(f"Starting download from URL: {url}")

            pdf_filename: str = URLPDFDownloader._extract_filename(url)
            logger.info(f"Extracted filename from URL: {pdf_filename}")

            local_path: str = URLPDFDownloader._download_and_validate(
                url, pdf_filename
            )
            logger.info(f"[SUCCESS] Successfully downloaded PDF to: {local_path}")

            return local_path, pdf_filename

        except ValueError as ve:
            logger.error(f"[ERROR] Validation error: {str(ve)}")
            raise

        except requests.RequestException as re:
            error_msg: str = f"Network error downloading PDF: {str(re)}"
            logger.error(f"[ERROR] {error_msg}")
            raise ValueError(error_msg) from re

        except IOError as ie:
            error_msg: str = f"I/O error saving PDF: {str(ie)}"
            logger.error(f"[ERROR] {error_msg}")
            raise ValueError(error_msg) from ie

        except Exception as e:
            error_msg: str = f"Unexpected error: {str(e)}"
            logger.error(f"[ERROR] {error_msg}")
            raise ValueError(error_msg) from e

    @staticmethod
    def _validate_url(url: str) -> None:
        """Validate URL format and scheme."""
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")

        url_lower: str = url.lower().strip()

        if not url_lower.startswith(("http://", "https://")):
            raise ValueError("URL must start with 'http://' or 'https://'")

        try:
            parsed = urlparse(url_lower)
            if not parsed.netloc:
                raise ValueError("URL must contain a valid domain")
        except Exception as e:
            raise ValueError(f"Invalid URL format: {str(e)}")

    @staticmethod
    def _extract_filename(url: str) -> str:
        """Extract filename from URL."""
        try:
            parsed = urlparse(url)
            path: str = parsed.path

            if path and "/" in path:
                filename: str = path.split("/")[-1]

                if "?" in filename:
                    filename = filename.split("?")[0]

                if filename:
                    if not filename.lower().endswith(".pdf"):
                        filename += ".pdf"
                    return filename

        except Exception as e:
            logger.warning(f"Could not extract filename from URL path: {str(e)}")

        import time

        timestamp: int = int(time.time() * 1000)
        return f"download_{timestamp}.pdf"

    @staticmethod
    def _download_and_validate(url: str, filename: str) -> str:
        """Download PDF from URL and validate it."""
        headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        temp_pdf_dir: str = os.path.join(
            URLPDFDownloader.TEMP_DIR, "domain_aware_rag_pdf_downloads"
        )
        os.makedirs(temp_pdf_dir, exist_ok=True)

        local_path: str = os.path.join(temp_pdf_dir, filename)

        try:
            logger.info(f"Sending HTTP request to: {url}")

            response: requests.Response = requests.get(
                url,
                headers=headers,
                timeout=URLPDFDownloader.REQUEST_TIMEOUT,
                stream=True,
                allow_redirects=True,
            )
            response.raise_for_status()

            content_type: str = response.headers.get("content-type", "").lower()
            logger.info(f"Response content-type: {content_type}")

            if "pdf" not in content_type and content_type not in [
                "application/octet-stream",
                "",
            ]:
                logger.warning(
                    f"Content-type '{content_type}' may not be PDF, but proceeding..."
                )

            total_size: int = 0

            chunks: Iterator[bytes] = response.iter_content(
                chunk_size=URLPDFDownloader.CHUNK_SIZE
            )

            with open(local_path, "wb") as f:
                for chunk in chunks:
                    if chunk:
                        chunk_len: int = len(chunk)
                        total_size += chunk_len

                        if total_size > URLPDFDownloader.MAX_FILE_SIZE:
                            os.remove(local_path)
                            raise ValueError(
                                f"Downloaded file exceeds maximum size of "
                                f"{URLPDFDownloader.MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
                            )

                        f.write(chunk)

            logger.info(f"Downloaded {total_size / 1024:.2f} KB")

            if not os.path.exists(local_path):
                raise IOError(f"Failed to save file to {local_path}")

            if os.path.getsize(local_path) == 0:
                os.remove(local_path)
                raise ValueError("Downloaded file is empty")

            URLPDFDownloader._validate_pdf_magic(local_path)
            return local_path

        except requests.RequestException:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise

    @staticmethod
    def _validate_pdf_magic(filepath: str) -> None:
        """Validate PDF file by checking magic bytes."""
        try:
            with open(filepath, "rb") as f:
                magic: bytes = f.read(4)

            if not magic.startswith(b"%PDF"):
                raise ValueError(
                    f"File does not appear to be a valid PDF (magic bytes: {magic}). "
                    "Ensure the URL points to an actual PDF file."
                )

        except IOError as e:
            raise ValueError(f"Cannot validate PDF file: {str(e)}")

    @staticmethod
    def cleanup(filepath: str) -> None:
        """Clean up temporary PDF file after processing."""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file: {filepath}")

        except Exception as e:
            logger.warning(
                f"Could not delete temporary file {filepath}: {str(e)}"
            )