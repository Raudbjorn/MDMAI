#!/usr/bin/env python3
"""Command-line utility to migrate legacy JSON files to scalable JSON Lines format.

This utility helps migrate from the old JSON array format to the new append-only
JSON Lines format for maximum scalability and performance.

Usage:
    python migrate_json_to_jsonl.py [--data-dir PATH] [--dry-run] [--backup]

Example:
    # Migrate all files in default data directory
    python migrate_json_to_jsonl.py
    
    # Dry run to see what would be migrated
    python migrate_json_to_jsonl.py --dry-run
    
    # Specify custom data directory
    python migrate_json_to_jsonl.py --data-dir ./custom_data
    
    # Create backups before migration
    python migrate_json_to_jsonl.py --backup
"""

import asyncio
import argparse
import json
import gzip
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class LegacyFileMigrator:
    """Utility class for migrating legacy JSON files to JSON Lines format."""
    
    def __init__(self, data_dir: str, create_backups: bool = False):
        self.data_dir = Path(data_dir)
        self.create_backups = create_backups
        
    async def scan_for_legacy_files(self) -> List[Dict[str, Any]]:
        """Scan directory for legacy JSON files that need migration."""
        legacy_files = []
        
        if not self.data_dir.exists():
            logger.warning("Data directory does not exist", dir=str(self.data_dir))
            return legacy_files
        
        # Look for .json files (legacy format)
        json_files = list(self.data_dir.glob("**/*.json"))
        
        for json_file in json_files:
            # Skip metadata and index files
            if json_file.name in ["metadata.json", "indices.json"]:
                continue
                
            try:
                # Check if it contains usage records in legacy format
                with open(json_file, 'r') as f:
                    content = f.read(200)  # Read first 200 chars
                    
                if content.strip().startswith('[') or '"records"' in content:
                    # Load full file to analyze
                    with open(json_file, 'r') as f:
                        try:
                            data = json.load(f)
                            
                            # Detect legacy usage tracking format
                            records = []
                            if isinstance(data, list):
                                records = data
                            elif isinstance(data, dict) and "records" in data:
                                records = data["records"]
                            
                            if records and self._looks_like_usage_record(records[0]):
                                file_info = {
                                    "path": str(json_file),
                                    "size_bytes": json_file.stat().st_size,
                                    "record_count": len(records),
                                    "format": "legacy_json_array",
                                    "created": datetime.fromtimestamp(
                                        json_file.stat().st_ctime
                                    ).isoformat()
                                }
                                legacy_files.append(file_info)
                                
                        except json.JSONDecodeError:
                            logger.debug(f"Skipping non-JSON file: {json_file}")
                            
            except Exception as e:
                logger.warning(f"Error scanning file {json_file}: {e}")
                
        return legacy_files
    
    def _looks_like_usage_record(self, record: Dict[str, Any]) -> bool:
        """Check if a record looks like a usage tracking record."""
        required_fields = {"request_id", "provider_type", "model", "timestamp"}
        return isinstance(record, dict) and required_fields.issubset(record.keys())
    
    async def migrate_file(self, file_path: str, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate a single legacy file to JSON Lines format."""
        migration_result = {
            "file_path": file_path,
            "success": False,
            "records_migrated": 0,
            "backup_created": None,
            "new_file_path": None,
            "error": None
        }
        
        try:
            path = Path(file_path)
            
            # Read legacy file
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Extract records
            records = []
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "records" in data:
                records = data["records"]
            else:
                migration_result["error"] = "No records found in file"
                return migration_result
            
            if dry_run:
                migration_result["success"] = True
                migration_result["records_migrated"] = len(records)
                return migration_result
            
            # Create backup if requested
            if self.create_backups:
                backup_path = path.with_suffix(path.suffix + '.backup')
                import shutil
                shutil.copy2(str(path), str(backup_path))
                migration_result["backup_created"] = str(backup_path)
                logger.info(f"Backup created: {backup_path}")
            
            # Convert to JSON Lines format
            jsonl_path = path.with_suffix('.jsonl')
            
            with open(jsonl_path, 'w') as f:
                for record in records:
                    # Add migration metadata
                    record_with_meta = {
                        "schema_version": "1.0.0",
                        "migrated_from_legacy": True,
                        "migrated_at": datetime.now().isoformat(),
                        **record
                    }
                    f.write(json.dumps(record_with_meta, separators=(',', ':')) + '\n')
            
            # Remove original file
            path.unlink()
            
            migration_result["success"] = True
            migration_result["records_migrated"] = len(records)
            migration_result["new_file_path"] = str(jsonl_path)
            
            logger.info(f"File migrated successfully: {path} -> {jsonl_path} ({len(records)} records)")
            
        except Exception as e:
            migration_result["error"] = str(e)
            logger.error(f"Failed to migrate file {file_path}: {e}")
        
        return migration_result
    
    async def migrate_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate all legacy files in the data directory."""
        migration_summary = {
            "start_time": datetime.now().isoformat(),
            "total_files_found": 0,
            "files_migrated": 0,
            "total_records_migrated": 0,
            "errors": [],
            "files_processed": [],
            "dry_run": dry_run
        }
        
        try:
            # Scan for legacy files
            legacy_files = await self.scan_for_legacy_files()
            migration_summary["total_files_found"] = len(legacy_files)
            
            if not legacy_files:
                logger.info("No legacy files found to migrate")
                return migration_summary
            
            logger.info(f"Found {len(legacy_files)} legacy files to migrate")
            
            # Migrate each file
            for file_info in legacy_files:
                result = await self.migrate_file(file_info["path"], dry_run=dry_run)
                migration_summary["files_processed"].append(result)
                
                if result["success"]:
                    migration_summary["files_migrated"] += 1
                    migration_summary["total_records_migrated"] += result["records_migrated"]
                else:
                    migration_summary["errors"].append(result["error"])
            
            migration_summary["end_time"] = datetime.now().isoformat()
            
            logger.info(f"Migration completed: {migration_summary}")
            return migration_summary
            
        except Exception as e:
            migration_summary["end_time"] = datetime.now().isoformat()
            migration_summary["errors"].append(f"Migration failed: {str(e)}")
            logger.error(f"Migration process failed: {e}")
            raise


def print_migration_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted migration summary."""
    print("\n" + "="*60)
    print("JSON TO JSON LINES MIGRATION SUMMARY")
    print("="*60)
    
    if summary["dry_run"]:
        print("🔍 DRY RUN MODE - No files were actually changed")
    
    print(f"📁 Data Directory: {summary.get('data_dir', 'N/A')}")
    print(f"⏰ Start Time: {summary['start_time']}")
    print(f"⏰ End Time: {summary.get('end_time', 'N/A')}")
    print(f"📊 Total Files Found: {summary['total_files_found']}")
    print(f"✅ Files Migrated: {summary['files_migrated']}")
    print(f"📝 Total Records Migrated: {summary['total_records_migrated']}")
    
    if summary["errors"]:
        print(f"❌ Errors: {len(summary['errors'])}")
        for error in summary["errors"][:3]:  # Show first 3 errors
            print(f"   • {error}")
        if len(summary["errors"]) > 3:
            print(f"   ... and {len(summary['errors']) - 3} more errors")
    
    print("\n📋 File Details:")
    for file_result in summary["files_processed"]:
        status = "✅" if file_result["success"] else "❌"
        records = file_result["records_migrated"]
        path = Path(file_result["file_path"]).name
        print(f"   {status} {path} ({records} records)")
    
    if not summary["dry_run"] and summary["files_migrated"] > 0:
        print("\n🎉 Migration completed successfully!")
        print("Your files are now using the scalable JSON Lines format.")
    elif summary["dry_run"]:
        print("\n💡 Run without --dry-run to perform the actual migration.")
    
    print("="*60)


async def main():
    """Main entry point for the migration utility."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy JSON files to scalable JSON Lines format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data/usage_tracking",
        help="Directory containing usage tracking data (default: ./data/usage_tracking)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    
    parser.add_argument(
        "--backup", 
        action="store_true",
        help="Create backup files before migration"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print("🚀 Starting JSON to JSON Lines migration...")
        print(f"📁 Data Directory: {args.data_dir}")
        print(f"🔍 Dry Run: {args.dry_run}")
        print(f"💾 Create Backups: {args.backup}")
        
        migrator = LegacyFileMigrator(args.data_dir, args.backup)
        summary = await migrator.migrate_all(dry_run=args.dry_run)
        summary["data_dir"] = args.data_dir
        
        print_migration_summary(summary)
        
        # Exit with error code if there were failures
        if summary["errors"]:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Migration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())