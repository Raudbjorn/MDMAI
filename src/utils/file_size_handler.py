"""File size handling and user confirmation utilities."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config.logging_config import get_logger

logger = get_logger(__name__)


class FileSizeCategory(Enum):
    """Categories for file sizes."""

    SMALL = "small"  # < 10MB
    MEDIUM = "medium"  # 10MB - 50MB
    LARGE = "large"  # 50MB - 100MB
    VERY_LARGE = "very_large"  # 100MB - 500MB
    EXCESSIVE = "excessive"  # > 500MB


class FileSizeHandler:
    """Handles file size validation and user confirmation."""

    # Size thresholds in bytes
    THRESHOLDS = {
        FileSizeCategory.SMALL: 10 * 1024 * 1024,  # 10MB
        FileSizeCategory.MEDIUM: 50 * 1024 * 1024,  # 50MB
        FileSizeCategory.LARGE: 100 * 1024 * 1024,  # 100MB
        FileSizeCategory.VERY_LARGE: 500 * 1024 * 1024,  # 500MB
    }

    # Estimated processing times (rough estimates)
    PROCESSING_TIME_ESTIMATES = {
        FileSizeCategory.SMALL: "less than 1 minute",
        FileSizeCategory.MEDIUM: "1-3 minutes",
        FileSizeCategory.LARGE: "3-5 minutes",
        FileSizeCategory.VERY_LARGE: "5-15 minutes",
        FileSizeCategory.EXCESSIVE: "15+ minutes",
    }

    @classmethod
    def categorize_file_size(cls, size_bytes: int) -> FileSizeCategory:
        """
        Categorize file size into predefined categories.

        Args:
            size_bytes: File size in bytes

        Returns:
            File size category
        """
        if size_bytes < cls.THRESHOLDS[FileSizeCategory.SMALL]:
            return FileSizeCategory.SMALL
        elif size_bytes < cls.THRESHOLDS[FileSizeCategory.MEDIUM]:
            return FileSizeCategory.MEDIUM
        elif size_bytes < cls.THRESHOLDS[FileSizeCategory.LARGE]:
            return FileSizeCategory.LARGE
        elif size_bytes < cls.THRESHOLDS[FileSizeCategory.VERY_LARGE]:
            return FileSizeCategory.VERY_LARGE
        else:
            return FileSizeCategory.EXCESSIVE

    @classmethod
    def format_file_size(cls, size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    @classmethod
    def get_file_info(cls, file_path: Path) -> Dict[str, Any]:
        """
        Get comprehensive file information.

        Args:
            file_path: Path to file

        Returns:
            File information dictionary
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        size_bytes = file_path.stat().st_size
        category = cls.categorize_file_size(size_bytes)

        return {
            "path": str(file_path),
            "name": file_path.name,
            "size_bytes": size_bytes,
            "size_formatted": cls.format_file_size(size_bytes),
            "category": category.value,
            "estimated_time": cls.PROCESSING_TIME_ESTIMATES[category],
            "requires_confirmation": category
            in [FileSizeCategory.LARGE, FileSizeCategory.VERY_LARGE, FileSizeCategory.EXCESSIVE],
            "warning_level": cls._get_warning_level(category),
        }

    @classmethod
    def _get_warning_level(cls, category: FileSizeCategory) -> str:
        """
        Get warning level for file size category.

        Args:
            category: File size category

        Returns:
            Warning level (none, info, warning, critical)
        """
        if category == FileSizeCategory.SMALL:
            return "none"
        elif category == FileSizeCategory.MEDIUM:
            return "info"
        elif category == FileSizeCategory.LARGE:
            return "warning"
        elif category == FileSizeCategory.VERY_LARGE:
            return "warning"
        else:  # EXCESSIVE
            return "critical"

    @classmethod
    def generate_confirmation_message(cls, file_info: Dict[str, Any]) -> str:
        """
        Generate a user-friendly confirmation message.

        Args:
            file_info: File information from get_file_info

        Returns:
            Confirmation message
        """
        category = FileSizeCategory(file_info["category"])

        if category == FileSizeCategory.LARGE:
            return (
                f"⚠️ **Large File Warning**\n\n"
                f"The file '{file_info['name']}' is {file_info['size_formatted']}, "
                f"which is considered large.\n\n"
                f"**Estimated processing time:** {file_info['estimated_time']}\n\n"
                f"Processing large files may:\n"
                f"- Take significant time to complete\n"
                f"- Use considerable system resources\n"
                f"- Potentially impact system performance\n\n"
                f"Would you like to continue with processing this file?"
            )
        elif category == FileSizeCategory.VERY_LARGE:
            return (
                f"⚠️ **Very Large File Warning**\n\n"
                f"The file '{file_info['name']}' is {file_info['size_formatted']}, "
                f"which is very large.\n\n"
                f"**Estimated processing time:** {file_info['estimated_time']}\n\n"
                f"⚠️ **Important considerations:**\n"
                f"- Processing will take considerable time\n"
                f"- High memory usage expected\n"
                f"- System may become less responsive\n"
                f"- Consider processing during off-peak hours\n\n"
                f"Are you sure you want to process this very large file?"
            )
        else:  # EXCESSIVE
            return (
                f"🔴 **Excessive File Size Warning**\n\n"
                f"The file '{file_info['name']}' is {file_info['size_formatted']}, "
                f"which exceeds recommended limits.\n\n"
                f"**Estimated processing time:** {file_info['estimated_time']}\n\n"
                f"⚠️ **Critical Warning:**\n"
                f"- Processing may fail due to memory constraints\n"
                f"- System may become unresponsive\n"
                f"- Consider splitting the file into smaller parts\n"
                f"- Recommended maximum size: 100MB\n\n"
                f"**This is not recommended.** Do you still want to attempt processing?"
            )

    @classmethod
    async def check_and_confirm(
        cls,
        file_path: Path,
        confirmation_callback: Optional[Any] = None,
        auto_approve_threshold: FileSizeCategory = FileSizeCategory.MEDIUM,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check file size and get confirmation if needed.

        Args:
            file_path: Path to file
            confirmation_callback: Optional callback for getting user confirmation
            auto_approve_threshold: Auto-approve files up to this category

        Returns:
            Tuple of (should_proceed, file_info)
        """
        file_info = cls.get_file_info(file_path)
        category = FileSizeCategory(file_info["category"])

        # Auto-approve small files
        if category.value <= auto_approve_threshold.value:
            logger.info(f"File size {file_info['size_formatted']} auto-approved")
            return True, file_info

        # Files requiring confirmation
        if file_info["requires_confirmation"]:
            confirmation_msg = cls.generate_confirmation_message(file_info)

            if confirmation_callback:
                # Use provided callback for confirmation
                confirmed = await confirmation_callback(confirmation_msg, file_info)
            else:
                # Log warning and return info for caller to handle
                logger.warning(
                    f"Large file requires confirmation: {file_info['name']} "
                    f"({file_info['size_formatted']})"
                )
                file_info["confirmation_message"] = confirmation_msg
                return False, file_info

            if confirmed:
                logger.info(f"User confirmed processing of large file: {file_info['name']}")
                return True, file_info
            else:
                logger.info(f"User declined processing of large file: {file_info['name']}")
                return False, file_info

        return True, file_info

    @classmethod
    def get_processing_recommendations(cls, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get processing recommendations based on file size.

        Args:
            file_info: File information from get_file_info

        Returns:
            Processing recommendations
        """
        category = FileSizeCategory(file_info["category"])

        recommendations = {
            "batch_size": 100,  # Default
            "use_parallel_processing": False,
            "cache_embeddings": True,
            "optimize_memory": False,
            "chunk_size": 1000,  # Default chunk size
        }

        if category == FileSizeCategory.MEDIUM:
            recommendations["batch_size"] = 50
            recommendations["use_parallel_processing"] = True

        elif category == FileSizeCategory.LARGE:
            recommendations["batch_size"] = 25
            recommendations["use_parallel_processing"] = True
            recommendations["optimize_memory"] = True
            recommendations["chunk_size"] = 800

        elif category in [FileSizeCategory.VERY_LARGE, FileSizeCategory.EXCESSIVE]:
            recommendations["batch_size"] = 10
            recommendations["use_parallel_processing"] = True
            recommendations["optimize_memory"] = True
            recommendations["chunk_size"] = 500
            recommendations["cache_embeddings"] = False  # Too memory intensive

        return recommendations
