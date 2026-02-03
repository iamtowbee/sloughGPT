#!/usr/bin/env python3
"""
SloughGPT User Management Demo
Demonstrates user authentication and authorization system
"""

import asyncio
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_user_management():
    """Demonstrate user management capabilities"""
    
    try:
        from sloughgpt.user_management import (
            UserManager, UserRole, Permission, get_user_manager, UserConfig
        )
    except ImportError as e:
        logger.error(f"User management not available: {e}")
        logger.info("Install user management dependencies with: pip install -r user-requirements.txt")
        return
        
    logger.info("👥 SloughGPT User Management Demo")
    logger.info("=" * 50)
    
    # Initialize user manager
    user_manager = get_user_manager()
    
    # Demo: Create users with different roles
    logger.info("🔧 Creating users with different roles...")
    
    try:
        # Create admin user
        admin_result = user_manager.create_user(
            username="admin",
            email="admin@sloughgpt.local",
            password="admin123456",
            role=UserRole.ADMIN,
            config=UserConfig(max_requests_per_hour=1000, cost_limit_monthly=1000.0)
        )
        logger.info(f"✅ Admin user created: {admin_result['user']['username']}")
        logger.info(f"   Permissions: {len(admin_result['permissions'])} permissions")
        
        # Create regular user
        user_result = user_manager.create_user(
            username="alice",
            email="alice@sloughgpt.local", 
            password="alice123456",
            role=UserRole.USER,
            config=UserConfig(max_requests_per_hour=100, cost_limit_monthly=50.0)
        )
        logger.info(f"✅ Regular user created: {user_result['user']['username']}")
        logger.info(f"   Permissions: {len(user_result['permissions'])} permissions")
        
        # Create moderator user
        moderator_result = user_manager.create_user(
            username="moderator",
            email="moderator@sloughgpt.local",
            password="mod123456", 
            role=UserRole.MODERATOR
        )
        logger.info(f"✅ Moderator user created: {moderator_result['user']['username']}")
        logger.info(f"   Permissions: {len(moderator_result['permissions'])} permissions")
        
    except Exception as e:
        logger.error(f"❌ Error creating users: {str(e)}")
        # Users might already exist
    
    # Demo: Authentication
    logger.info("\n🔐 Testing authentication...")
    
    try:
        auth_result = user_manager.authenticate_user("alice", "alice123456")
        if auth_result:
            logger.info(f"✅ Authentication successful for {auth_result['user']['username']}")
            logger.info(f"   Access token: {auth_result['access_token'][:30]}...")
            logger.info(f"   Permissions: {auth_result['permissions']}")
            
            # Test permission checking
            user_permissions = set(auth_result['permissions'])
            can_infer = Permission.MODEL_INFERENCE.value in user_permissions
            can_admin = Permission.SYSTEM_ADMIN.value in user_permissions
            
            logger.info(f"   Can use model inference: {can_infer}")
            logger.info(f"   Can administer system: {can_admin}")
            
        else:
            logger.error("❌ Authentication failed")
            
    except Exception as e:
        logger.error(f"❌ Authentication error: {str(e)}")
    
    # Demo: API Key Management
    logger.info("\n🔑 Testing API key management...")
    
    try:
        # Get user ID for alice
        auth_result = user_manager.authenticate_user("alice", "alice123456")
        if auth_result:
            user_id = auth_result['user']['id']
            
            # Create API key
            api_key_result = user_manager.create_api_key(
                user_id=user_id,
                name="Demo API Key",
                permissions=[Permission.MODEL_INFERENCE.value, Permission.DATA_READ.value],
                rate_limit=50,
                expires_in_days=30
            )
            
            logger.info(f"✅ API key created: {api_key_result['api_key'][:20]}...")
            logger.info(f"   Name: {api_key_result['key_record']['name']}")
            logger.info(f"   Rate limit: {api_key_result['key_record']['rate_limit']}/hour")
            
            # Test API key authentication
            api_key = api_key_result['api_key']
            api_auth = user_manager.authenticate_api_key(api_key)
            
            if api_auth:
                logger.info(f"✅ API key authentication successful")
                logger.info(f"   User: {api_auth['user']['username']}")
                logger.info(f"   Permissions: {api_auth['permissions']}")
            
            # Revoke API key
            revoke_success = user_manager.revoke_api_key(
                api_key_result['key_record']['id'], 
                user_id
            )
            logger.info(f"✅ API key revoked: {revoke_success}")
            
    except Exception as e:
        logger.error(f"❌ API key error: {str(e)}")
    
    # Demo: User Statistics
    logger.info("\n📊 Getting user statistics...")
    
    try:
        # Get stats for alice
        auth_result = user_manager.authenticate_user("alice", "alice123456")
        if auth_result:
            user_id = auth_result['user']['id']
            stats = user_manager.get_user_stats(user_id)
            
            logger.info(f"👤 User: {stats['user']['username']}")
            logger.info(f"   Role: {stats['user']['role']}")
            logger.info(f"   Active since: {stats['user']['created_at']}")
            logger.info(f"   Last login: {stats['user']['last_login']}")
            logger.info(f"   API keys: {len(stats['api_keys'])}")
            logger.info(f"   Active sessions: {len(stats['active_sessions'])}")
            logger.info(f"   Usage this hour: {stats['usage_summary']['requests_this_hour']} requests")
            logger.info(f"   Cost this month: ${stats['usage_summary']['cost_this_month']:.2f}")
            
    except Exception as e:
        logger.error(f"❌ Stats error: {str(e)}")
    
    # Demo: Permission System
    logger.info("\n🔒 Testing permission system...")
    
    try:
        # Show permissions for each role
        for role in UserRole:
            permissions = user_manager.get_user_permissions(role)
            permission_names = [p.value for p in permissions]
            
            logger.info(f"\n📋 Role: {role.value}")
            logger.info(f"   Permissions: {len(permissions)} total")
            
            # Show key permissions
            key_perms = ['model:inference', 'model:train', 'user:admin', 'system:admin']
            for perm in key_perms:
                has_perm = perm in permission_names
                status = "✅" if has_perm else "❌"
                logger.info(f"   {status} {perm}")
                
    except Exception as e:
        logger.error(f"❌ Permission error: {str(e)}")
    
    logger.info("\n🎉 User Management Demo Complete!")
    logger.info("\n💡 Try these CLI commands:")
    logger.info("   python3 sloughgpt/user_management.py create-user \\")
    logger.info("       --username testuser --email test@example.com --password pass123")
    logger.info("   python3 sloughgpt/user_management.py authenticate \\")
    logger.info("       --username testuser --password pass123")
    logger.info("   python3 sloughgpt/user_management.py create-api-key \\")
    logger.info("       --username testuser --password pass123 --api-key-name 'My Key'")

def demo_advanced_features():
    """Demonstrate advanced user management features"""
    
    logger.info("\n🚀 Advanced User Management Features")
    logger.info("=" * 40)
    
    try:
        from sloughgpt.user_management import get_user_manager, UserRole, Permission
        
        user_manager = get_user_manager()
        
        # Demo: Password reset
        logger.info("🔄 Testing password reset...")
        reset_token = user_manager.reset_password("alice@sloughgpt.local")
        logger.info(f"✅ Password reset token generated: {reset_token[:20]}...")
        
        # Demo: User update
        logger.info("\n📝 Testing user update...")
        auth_result = user_manager.authenticate_user("alice", "alice123456")
        if auth_result:
            user_id = auth_result['user']['id']
            updates = {
                "config": {
                    "max_requests_per_hour": 200,
                    "cost_limit_monthly": 75.0,
                    "default_model": "sloughgpt-medium"
                }
            }
            
            updated_user = user_manager.update_user(user_id, updates)
            if updated_user:
                logger.info("✅ User updated successfully")
                logger.info(f"   New request limit: {updated_user['config']['max_requests_per_hour']}/hour")
                logger.info(f"   New cost limit: ${updated_user['config']['cost_limit_monthly']}/month")
        
        # Demo: Change password
        logger.info("\n🔐 Testing password change...")
        auth_result = user_manager.authenticate_user("alice", "alice123456")
        if auth_result:
            user_id = auth_result['user']['id']
            change_success = user_manager.change_password(
                user_id, "alice123456", "newpassword123"
            )
            logger.info(f"✅ Password change successful: {change_success}")
            
            # Test new password
            new_auth = user_manager.authenticate_user("alice", "newpassword123")
            logger.info(f"✅ Authentication with new password: {bool(new_auth)}")
            
    except Exception as e:
        logger.error(f"❌ Advanced features error: {str(e)}")

if __name__ == "__main__":
    print("👥 SloughGPT User Management Demo")
    print("=" * 50)
    
    # Run basic demo
    demo_user_management()
    
    # Run advanced demo
    demo_advanced_features()
    
    print("\n" + "=" * 50)
    print("📦 Install user management dependencies:")
    print("   pip install -r user-requirements.txt")
    print("\n🔒 Security Features:")
    print("   • JWT-based authentication")
    print("   • API key management")
    print("   • Role-based permissions")
    print("   • Rate limiting")
    print("   • Password encryption")
    print("   • Session management")