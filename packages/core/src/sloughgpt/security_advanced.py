"""Advanced security features for SloughGPT."""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import secrets
import asyncio
import logging
from enum import Enum

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    import base64
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

try:
    import pyotp
    import qrcode
    from PIL import Image
    import io
    HAS_2FA = True
except ImportError:
    HAS_2FA = False


class SecurityLevel(Enum):
    """Security level classifications."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat level for security events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event structure."""
    event_id: str
    event_type: str
    threat_level: ThreatLevel
    user_id: Optional[int]
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enabled: bool = True
    created_at: datetime
    updated_at: datetime


@dataclass
class EncryptionKey:
    """Encryption key information."""
    key_id: str
    algorithm: str
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool = True


class SecurityManager:
    """Advanced security manager for enterprise features."""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.policies: Dict[str, SecurityPolicy] = {}
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.2fa_secrets: Dict[int, Dict[str, str]] = {}  # user_id -> secret, backup
        self.session_data: Dict[str, Dict[str, Any]] = {}
        self.master_key = None
        self.initialized = False
    
    async def initialize(self, master_key: Optional[bytes] = None) -> bool:
        """Initialize security manager."""
        try:
            # Generate master key if not provided
            if master_key is None:
                master_key = secrets.token_bytes(32)
            
            self.master_key = master_key
            
            # Initialize encryption key
            if HAS_CRYPTOGRAPHY:
                self.encryption_keys["master"] = EncryptionKey(
                    key_id="master",
                    algorithm="AES-256",
                    key_data=master_key,
                    created_at=datetime.now()
                )
            
            # Create default security policies
            await self._create_default_policies()
            
            self.initialized = True
            logging.info("Security manager initialized")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize security manager: {e}")
            return False
    
    def generate_session_token(self, user_id: int, permissions: List[str],
                            expires_hours: int = 24) -> str:
        """Generate secure session token."""
        if not self.initialized:
            raise RuntimeError("Security manager not initialized")
        
        # Create session data
        session_data = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=expires_hours)).isoformat()
        }
        
        # Generate session ID
        session_id = secrets.token_urlsafe(32)
        
        # Store session data
        self.session_data[session_id] = session_data
        
        # Create signed token
        token_data = f"{session_id}:{user_id}:{datetime.now().timestamp()}"
        signature = self._sign_data(token_data.encode())
        
        return base64.b64encode(f"{token_data}:{signature.decode()}".encode()).decode()
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode session token."""
        if not self.initialized:
            return None
        
        try:
            # Decode token
            token_bytes = base64.b64decode(token.encode())
            token_data, signature = token_bytes.decode().split(":", 1)
            
            # Verify signature
            if not self._verify_signature(token_data.encode(), signature.encode()):
                return None
            
            # Parse token data
            session_id, user_id, timestamp = token_data.split(":", 2)
            
            # Check if session exists
            if session_id not in self.session_data:
                return None
            
            # Check if session has expired
            expires_at = datetime.fromisoformat(self.session_data[session_id]["expires_at"])
            if datetime.now() > expires_at:
                del self.session_data[session_id]
                return None
            
            return self.session_data[session_id]
            
        except Exception as e:
            logging.error(f"Session token validation error: {e}")
            return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        if session_id in self.session_data:
            del self.session_data[session_id]
            logging.info(f"Session invalidated: {session_id}")
            return True
        return False
    
    def encrypt_data(self, data: str, key_id: str = "master") -> Optional[str]:
        """Encrypt data with AES encryption."""
        if not HAS_CRYPTOGRAPHY or not self.initialized:
            return None
        
        if key_id not in self.encryption_keys:
            return None
        
        try:
            key = self.encryption_keys[key_id].key_data
            cipher_suite = Fernet(self._derive_key(key))
            
            encrypted_data = cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            logging.error(f"Encryption error: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: str, key_id: str = "master") -> Optional[str]:
        """Decrypt AES encrypted data."""
        if not HAS_CRYPTOGRAPHY or not self.initialized:
            return None
        
        if key_id not in self.encryption_keys:
            return None
        
        try:
            key = self.encryption_keys[key_id].key_data
            cipher_suite = Fernet(self._derive_key(key))
            
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = cipher_suite.decrypt(encrypted_bytes)
            
            return decrypted_data.decode()
            
        except Exception as e:
            logging.error(f"Decryption error: {e}")
            return None
    
    def _derive_key(self, key_material: bytes) -> bytes:
        """Derive encryption key from key material."""
        salt = b"sloughgpt_salt"  # In production, use proper salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(key_material)
    
    def _sign_data(self, data: bytes) -> bytes:
        """Sign data with HMAC."""
        if not self.master_key:
            return b""
        
        h = hashlib.hmac.new(self.master_key, data, hashlib.sha256).digest()
        return base64.b64encode(h).decode()
    
    def _verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify HMAC signature."""
        if not self.master_key:
            return False
        
        expected_signature = self._sign_data(data)
        return secrets.compare_digest(signature, expected_signature.encode())
    
    async def create_2fa_secret(self, user_id: int) -> Optional[str]:
        """Create 2FA secret for user."""
        if not HAS_2FA:
            return None
        
        secret = pyotp.random_base32()
        backup_codes = [secrets.token_hex(4) for _ in range(10)]
        
        self.2fa_secrets[user_id] = {
            "secret": secret,
            "backup_codes": backup_codes,
            "created_at": datetime.now().isoformat()
        }
        
        return secret
    
    def generate_2fa_qr_code(self, secret: str, user_email: str) -> Optional[str]:
        """Generate QR code for 2FA setup."""
        if not HAS_2FA:
            return None
        
        try:
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=user_email,
                issuer_name="SloughGPT"
            )
            
            qr = qrcode.QRCode(version=1)
            qr.add_data(totp_uri)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logging.error(f"QR code generation error: {e}")
            return None
    
    def verify_2fa_token(self, user_id: int, token: str) -> bool:
        """Verify 2FA token."""
        if not HAS_2FA or user_id not in self.2fa_secrets:
            return False
        
        secret = self.2fa_secrets[user_id]["secret"]
        totp = pyotp.TOTP(secret)
        
        # Check TOTP token
        if totp.verify(token):
            return True
        
        # Check backup codes
        backup_codes = self.2fa_secrets[user_id]["backup_codes"]
        if token in backup_codes:
            backup_codes.remove(token)
            return True
        
        return False
    
    async def check_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Check IP reputation and threat intelligence."""
        # Basic IP analysis - in production, integrate with threat intelligence APIs
        ip_info = {
            "ip_address": ip_address,
            "is_blocked": ip_address in self.blocked_ips,
            "is_suspicious": ip_address in self.suspicious_ips,
            "reputation_score": 80,  # Default score
            "country": "Unknown",
            "is_tor_exit": False,
            "is_proxy": False
        }
        
        # Check suspicious activity
        if ip_address in self.suspicious_ips:
            ip_info["suspicious_reason"] = self.suspicious_ips[ip_address].get("reason", "Unknown")
            ip_info["suspicious_count"] = self.suspicious_ips[ip_address].get("count", 0)
        
        # Basic IP validation
        if self._is_private_ip(ip_address):
            ip_info["is_private"] = True
        else:
            ip_info["is_private"] = False
        
        return ip_info
    
    def _is_private_ip(self, ip_address: str) -> bool:
        """Check if IP is private."""
        try:
            # Simple private IP ranges
            private_ranges = [
                ("10.0.0.0", "10.255.255.255"),
                ("172.16.0.0", "172.31.255.255"),
                ("192.168.0.0", "192.168.255.255"),
                ("127.0.0.0", "127.255.255.255"),
            ]
            
            ip_int = sum(int(octet) * (256 ** (3 - i)) for i, octet in enumerate(ip_address.split('.')))
            
            for start, end in private_ranges:
                start_int = sum(int(octet) * (256 ** (3 - i)) for i, octet in enumerate(start.split('.')))
                end_int = sum(int(octet) * (256 ** (3 - i)) for i, octet in enumerate(end.split('.')))
                
                if start_int <= ip_int <= end_int:
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def check_rate_limit(self, ip_address: str, action: str,
                           window_seconds: int = 60, max_requests: int = 100) -> Dict[str, Any]:
        """Check rate limiting for IP and action."""
        key = f"{ip_address}:{action}"
        current_time = datetime.now()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                "requests": [],
                "blocked_until": None
            }
        
        rate_limit_data = self.rate_limits[key]
        
        # Check if currently blocked
        if rate_limit_data["blocked_until"] and current_time < rate_limit_data["blocked_until"]:
            return {
                "allowed": False,
                "reason": "Rate limit exceeded",
                "blocked_until": rate_limit_data["blocked_until"].isoformat(),
                "retry_after": int((rate_limit_data["blocked_until"] - current_time).total_seconds())
            }
        
        # Clean up old requests
        cutoff_time = current_time - timedelta(seconds=window_seconds)
        rate_limit_data["requests"] = [
            req_time for req_time in rate_limit_data["requests"]
            if req_time > cutoff_time
        ]
        
        # Check if limit exceeded
        if len(rate_limit_data["requests"]) >= max_requests:
            rate_limit_data["blocked_until"] = current_time + timedelta(minutes=5)
            
            # Log security event
            await self.log_security_event(
                event_type="rate_limit_exceeded",
                threat_level=ThreatLevel.MEDIUM,
                ip_address=ip_address,
                details={
                    "action": action,
                    "requests_count": len(rate_limit_data["requests"]),
                    "window_seconds": window_seconds,
                    "max_allowed": max_requests,
                    "blocked_until": rate_limit_data["blocked_until"].isoformat()
                }
            )
            
            return {
                "allowed": False,
                "reason": "Rate limit exceeded",
                "blocked_until": rate_limit_data["blocked_until"].isoformat(),
                "retry_after": 300
            }
        
        # Add current request
        rate_limit_data["requests"].append(current_time)
        
        return {
            "allowed": True,
            "remaining_requests": max_requests - len(rate_limit_data["requests"]),
            "reset_time": (current_time + timedelta(seconds=window_seconds)).isoformat()
        }
    
    async def log_security_event(self, event_type: str, threat_level: ThreatLevel,
                             user_id: Optional[int], ip_address: str,
                             user_agent: str = "", details: Optional[Dict[str, Any]] = None) -> str:
        """Log a security event."""
        event_id = str(uuid.uuid4())
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            threat_level=threat_level,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            timestamp=datetime.now()
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 10000)
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]
        
        # Trigger automated responses for critical events
        if threat_level == ThreatLevel.CRITICAL:
            await self._handle_critical_event(event)
        
        logging.warning(f"Security event logged: {event_type} - {threat_level.value}")
        return event_id
    
    async def _handle_critical_event(self, event: SecurityEvent):
        """Handle critical security events."""
        # Block suspicious IP temporarily
        if event.ip_address not in self.blocked_ips:
            self.suspicious_ips[event.ip_address] = {
                "count": self.suspicious_ips.get(event.ip_address, {}).get("count", 0) + 1,
                "reason": "Critical security event",
                "last_event": event.timestamp
            }
        
        # Auto-block if multiple critical events from same IP
        ip_data = self.suspicious_ips.get(event.ip_address, {})
        if ip_data.get("count", 0) >= 3:
            self.blocked_ips.add(event.ip_address)
            logging.critical(f"IP auto-blocked: {event.ip_address}")
    
    async def create_security_policy(self, policy_id: str, name: str, description: str,
                               rules: List[Dict[str, Any]]) -> bool:
        """Create a security policy."""
        policy = SecurityPolicy(
            policy_id=policy_id,
            name=name,
            description=description,
            rules=rules,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.policies[policy_id] = policy
        logging.info(f"Security policy created: {policy_id}")
        return True
    
    async def evaluate_security_policy(self, policy_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate security policy against context."""
        if policy_id not in self.policies:
            return {"allowed": False, "reason": "Policy not found"}
        
        policy = self.policies[policy_id]
        
        if not policy.enabled:
            return {"allowed": False, "reason": "Policy disabled"}
        
        # Evaluate all rules
        for rule in policy.rules:
            result = await self._evaluate_rule(rule, context)
            if not result["passed"]:
                return {
                    "allowed": False,
                    "rule": rule,
                    "reason": result["reason"]
                }
        
        return {"allowed": True, "policy": policy_id}
    
    async def _evaluate_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate individual security rule."""
        rule_type = rule.get("type")
        
        if rule_type == "time_based":
            # Time-based rules
            start_time = rule.get("start_time")
            end_time = rule.get("end_time")
            current_time = datetime.now().time()
            
            if start_time and end_time:
                if not (start_time <= current_time <= end_time):
                    return {"passed": False, "reason": "Outside allowed time window"}
        
        elif rule_type == "ip_whitelist":
            # IP whitelist rules
            allowed_ips = rule.get("allowed_ips", [])
            ip_address = context.get("ip_address")
            
            if ip_address and ip_address not in allowed_ips:
                return {"passed": False, "reason": "IP not in whitelist"}
        
        elif rule_type == "geolocation":
            # Geolocation rules
            allowed_countries = rule.get("allowed_countries", [])
            blocked_countries = rule.get("blocked_countries", [])
            
            # In production, use actual geolocation service
            country = context.get("country", "Unknown")
            
            if allowed_countries and country not in allowed_countries:
                return {"passed": False, "reason": f"Country not allowed: {country}"}
            
            if country in blocked_countries:
                return {"passed": False, "reason": f"Country blocked: {country}"}
        
        elif rule_type == "user_role":
            # User role rules
            allowed_roles = rule.get("allowed_roles", [])
            user_role = context.get("user_role")
            
            if user_role not in allowed_roles:
                return {"passed": False, "reason": f"Role not allowed: {user_role}"}
        
        return {"passed": True}
    
    async def _create_default_policies(self):
        """Create default security policies."""
        # Office hours policy
        await self.create_security_policy(
            policy_id="office_hours",
            name="Office Hours Access",
            description="Restrict access to business hours",
            rules=[
                {
                    "type": "time_based",
                    "start_time": "09:00",
                    "end_time": "17:00",
                    "timezone": "UTC"
                }
            ]
        )
        
        # Admin-only operations policy
        await self.create_security_policy(
            policy_id="admin_operations",
            name="Admin Operations Only",
            description="Restrict sensitive operations to admin users",
            rules=[
                {
                    "type": "user_role",
                    "allowed_roles": ["admin"]
                }
            ]
        )
        
        # IP whitelist policy
        await self.create_security_policy(
            policy_id="internal_network",
            name="Internal Network Access",
            description="Restrict access to internal network IPs",
            rules=[
                {
                    "type": "ip_whitelist",
                    "allowed_ips": ["192.168.1.0/24", "10.0.0.0/8"]
                }
            ]
        )
    
    async def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics and metrics."""
        recent_events = [
            event for event in self.security_events
            if event.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        stats = {
            "total_events": len(self.security_events),
            "recent_events_24h": len(recent_events),
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "active_sessions": len(self.session_data),
            "2fa_enabled_users": len(self.2fa_secrets),
            "active_policies": len([p for p in self.policies.values() if p.enabled]),
            "threat_breakdown": {},
            "top_threat_ips": []
        }
        
        # Threat level breakdown
        threat_counts = {}
        for event in recent_events:
            threat = event.threat_level.value
            threat_counts[threat] = threat_counts.get(threat, 0) + 1
        
        stats["threat_breakdown"] = threat_counts
        
        # Top threat IPs
        ip_threat_counts = {}
        for event in recent_events:
            ip_threat_counts[event.ip_address] = ip_threat_counts.get(event.ip_address, 0) + 1
        
        stats["top_threat_ips"] = sorted(
            ip_threat_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return stats
    
    async def cleanup_old_data(self, days: int = 90) -> Dict[str, int]:
        """Clean up old security data."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean old events
        old_events = len([
            event for event in self.security_events
            if event.timestamp < cutoff_date
        ])
        
        self.security_events = [
            event for event in self.security_events
            if event.timestamp >= cutoff_date
        ]
        
        # Clean expired sessions
        expired_sessions = 0
        for session_id, session_data in list(self.session_data.items()):
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if expires_at < datetime.now():
                del self.session_data[session_id]
                expired_sessions += 1
        
        # Clean old rate limit data
        current_time = datetime.now()
        old_rate_limits = 0
        
        for key, rate_data in list(self.rate_limits.items()):
            cutoff_time = current_time - timedelta(hours=1)
            rate_data["requests"] = [
                req_time for req_time in rate_data["requests"]
                if req_time > cutoff_time
            ]
            
            if not rate_data["requests"] and not rate_data.get("blocked_until"):
                del self.rate_limits[key]
                old_rate_limits += 1
        
        return {
            "old_events_removed": old_events,
            "expired_sessions_removed": expired_sessions,
            "old_rate_limits_removed": old_rate_limits,
            "cleanup_date": cutoff_date.isoformat()
        }


# Global security manager
security_manager = SecurityManager()