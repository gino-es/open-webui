#!/usr/bin/env python3
"""
Standalone CLI script for migrating chat data from JSON format to chat_message table.
This script can be run independently to migrate chat data without starting the full web server.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import open_webui modules
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Set up environment variables if not already set
if not os.getenv("WEBUI_SECRET_KEY"):
    key_file = SCRIPT_DIR / ".webui_secret_key"
    if key_file.exists():
        with open(key_file, "r") as f:
            os.environ["WEBUI_SECRET_KEY"] = f.read().strip()
    else:
        # Generate a temporary key for the migration
        import secrets
        os.environ["WEBUI_SECRET_KEY"] = secrets.token_urlsafe(32)

# Import after setting up environment
from open_webui.services.chat_migration_service import chat_migration_service
from open_webui.utils.logger import start_logger

def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Start the application logger
    start_logger()

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Migrate chat data from JSON format to chat_message table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check migration status
  python migrate_chats.py --status
  
  # Run migration with default settings
  python migrate_chats.py --migrate
  
  # Run migration with verbose logging
  python migrate_chats.py --migrate --verbose
  
  # Run migration with custom batch size
  python migrate_chats.py --migrate --batch-size 100
        """
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check migration status without running migration"
    )
    
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Run the migration process"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chats to process in each batch (default: 50)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually doing it"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.status and not args.migrate:
        parser.error("Please specify either --status or --migrate")
    
    try:
        # Check status
        if args.status:
            logger.info("Checking migration status...")
            status = chat_migration_service.check_migration_status()
            
            print("\n" + "="*60)
            print("MIGRATION STATUS")
            print("="*60)
            print(f"Total chats in database: {status['chat_table_count']}")
            print(f"Total chat messages: {status['chat_message_table_count']}")
            print(f"Chats with messages to migrate: {status['chats_with_messages']}")
            print(f"Chats already processed: {status['processed_chats']}")
            print(f"Completion percentage: {status['completion_percentage']}%")
            print()
            
            if status['migration_needed']:
                print("�� Migration needed - no messages have been migrated yet")
            elif status['partially_migrated']:
                print("⏳ Migration partially complete - some chats still need migration")
            elif status['migration_complete']:
                print("✅ Migration complete - all chats have been migrated")
            else:
                print("ℹ️  No migration needed - no chats with messages found")
            
            print("="*60)
        
        # Run migration
        if args.migrate:
            if args.dry_run:
                logger.info("DRY RUN MODE - No actual changes will be made")
                # For dry run, we'll just check status and show what would be done
                status = chat_migration_service.check_migration_status()
                if status['migration_needed'] or status['partially_migrated']:
                    print(f"\nWould migrate {status['chats_with_messages'] - status['processed_chats']} chats")
                else:
                    print("\nNo migration needed")
                return
            
            # Set custom batch size if specified
            if args.batch_size != 50:
                chat_migration_service.batch_size = args.batch_size
                logger.info(f"Using custom batch size: {args.batch_size}")
            
            logger.info("Starting chat migration...")
            result = chat_migration_service.migrate_chats()
            
            print("\n" + "="*60)
            print("MIGRATION RESULTS")
            print("="*60)
            print(f"Total chats processed: {result['total_chats']}")
            print(f"Messages migrated: {result['total_messages_migrated']}")
            print(f"Analytics chats skipped: {result['skipped_analytics_chats']}")
            print(f"Errors encountered: {result['errors']}")
            print(f"Migration successful: {result['success']}")
            print(f"Migration complete: {result['migration_complete']}")
            
            if result['success']:
                print("\n✅ Migration completed successfully!")
            else:
                print(f"\n⚠️  Migration completed with {result['errors']} errors")
            
            print("="*60)
    
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 