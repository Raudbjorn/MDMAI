#!/usr/bin/env python3
"""
Test script to verify all code review fixes have been properly implemented.
This tests all 14 identified issues from the code review.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Dict

# Project root
PROJECT_ROOT = Path(__file__).parent


class CodeReviewTester:
    """Test all code review fixes."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def test(self, name: str, condition: bool, details: str = ""):
        """Record a test result."""
        if condition:
            self.passed += 1
            print(f"✅ {name}")
            if details:
                print(f"   {details}")
        else:
            self.failed += 1
            print(f"❌ {name}")
            if details:
                print(f"   {details}")
        self.results.append((name, condition, details))
    
    def run_all_tests(self):
        """Run all code review fix tests."""
        print("=" * 80)
        print("TESTING CODE REVIEW FIXES")
        print("=" * 80)
        print()
        
        # 1. Test Dockerfile multi-stage build fix
        self.test_dockerfile_fix()
        
        # 2. Test security settings configuration fix
        self.test_security_settings_fix()
        
        # 3. Test main_secured.py removal
        self.test_main_secured_removal()
        
        # 4. Test cross-platform compatibility
        self.test_cross_platform_compatibility()
        
        # 5. Test sed portability in install.sh
        self.test_sed_portability()
        
        # 6. Test version parsing with packaging.version
        self.test_version_parsing()
        
        # 7. Test specific exception handling
        self.test_exception_handling()
        
        # 8. Test nginx.conf exists
        self.test_nginx_config()
        
        # 9. Test backup manager --delete option
        self.test_backup_delete_option()
        
        # 10. Test backup consistency (uses same system)
        self.test_backup_consistency()
        
        # 11. Test Makefile improvements
        self.test_makefile_improvements()
        
        # 12. Test Docker layer optimization
        self.test_docker_optimization()
        
        # 13. Test setup.py version extraction
        self.test_version_extraction()
        
        # 14. Test requirements-cpu.txt removal
        self.test_requirements_cleanup()
        
        # Print summary
        print()
        print("=" * 80)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("=" * 80)
        
        return self.failed == 0
    
    def test_dockerfile_fix(self):
        """Test Dockerfile multi-stage build fix."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        if not dockerfile.exists():
            self.test("Dockerfile exists", False)
            return
        
        content = dockerfile.read_text()
        # Check for correct COPY from builder stage
        correct_copy = "COPY --from=builder --chown=ttrpg:ttrpg /build/ /app/" in content
        self.test(
            "Dockerfile uses correct multi-stage COPY",
            correct_copy,
            "Line 76: COPY --from=builder --chown=ttrpg:ttrpg /build/ /app/"
        )
    
    def test_security_settings_fix(self):
        """Test security settings configuration fix."""
        main_py = PROJECT_ROOT / "src" / "main.py"
        if not main_py.exists():
            self.test("src/main.py exists", False)
            return
        
        content = main_py.read_text()
        # Check that we're not using getattr for security settings
        has_bad_getattr = "getattr(settings, 'security_" in content
        uses_direct_access = "settings.enable_authentication" in content
        
        self.test(
            "Security settings use direct attribute access",
            not has_bad_getattr and uses_direct_access,
            "Using settings.enable_authentication instead of getattr"
        )
    
    def test_main_secured_removal(self):
        """Test that main_secured.py has been removed."""
        main_secured = PROJECT_ROOT / "src" / "main_secured.py"
        self.test(
            "main_secured.py removed (dead code)",
            not main_secured.exists(),
            "File no longer exists"
        )
    
    def test_cross_platform_compatibility(self):
        """Test cross-platform utility module."""
        platform_utils = PROJECT_ROOT / "deploy" / "utils" / "platform_utils.py"
        self.test(
            "platform_utils.py created",
            platform_utils.exists(),
            "Cross-platform utility module exists"
        )
        
        if platform_utils.exists():
            content = platform_utils.read_text()
            has_windows = "is_windows" in content
            has_linux = "is_linux" in content
            has_macos = "is_macos" in content
            has_manage_service = "manage_service" in content
            
            self.test(
                "Platform detection functions",
                all([has_windows, has_linux, has_macos]),
                "is_windows(), is_linux(), is_macos() functions present"
            )
            
            self.test(
                "Cross-platform service management",
                has_manage_service,
                "manage_service() function handles systemctl, launchctl, sc"
            )
        
        # Check restore_manager uses platform utils
        restore_manager = PROJECT_ROOT / "deploy" / "backup" / "restore_manager.py"
        if restore_manager.exists():
            content = restore_manager.read_text()
            imports_utils = "from utils.platform_utils import" in content
            self.test(
                "restore_manager.py uses platform_utils",
                imports_utils,
                "Imports cross-platform utilities"
            )
    
    def test_sed_portability(self):
        """Test sed portability in install.sh."""
        install_sh = PROJECT_ROOT / "deploy" / "scripts" / "install.sh"
        if not install_sh.exists():
            self.test("install.sh exists", False)
            return
        
        content = install_sh.read_text()
        # Check for macOS-specific sed handling
        has_macos_sed = 'sed -i \'\'' in content
        has_os_check = '"$OS_TYPE" == "macos"' in content
        
        self.test(
            "install.sh handles sed portability",
            has_macos_sed and has_os_check,
            "Different sed syntax for macOS vs Linux"
        )
    
    def test_version_parsing(self):
        """Test version parsing uses packaging.version."""
        version_manager = PROJECT_ROOT / "deploy" / "migration" / "version_manager.py"
        if not version_manager.exists():
            self.test("version_manager.py exists", False)
            return
        
        content = version_manager.read_text()
        uses_packaging = "from packaging import version" in content or "from packaging.version import" in content
        no_regex_version = not ("re.match" in content and "version" in content and r"\d+\.\d+\.\d+" in content)
        
        self.test(
            "version_manager.py uses packaging.version",
            uses_packaging,
            "Using standard packaging library for version parsing"
        )
    
    def test_exception_handling(self):
        """Test specific exception handling."""
        files_to_check = [
            PROJECT_ROOT / "deploy" / "backup" / "restore_manager.py",
            PROJECT_ROOT / "deploy" / "backup" / "backup_manager.py"
        ]
        
        for file_path in files_to_check:
            if not file_path.exists():
                continue
            
            content = file_path.read_text()
            # Count broad vs specific exceptions
            broad_count = content.count("except Exception")
            specific_patterns = [
                "except IOError",
                "except OSError",
                "except json.JSONDecodeError",
                "except tarfile.TarError",
                "except ImportError",
                "except PermissionError",
                "except TypeError"
            ]
            
            has_specific = any(pattern in content for pattern in specific_patterns)
            
            self.test(
                f"{file_path.name} uses specific exceptions",
                has_specific and broad_count < 3,  # Allow a few broad catches for truly unexpected errors
                f"Using specific exception types like IOError, OSError, JSONDecodeError"
            )
    
    def test_nginx_config(self):
        """Test nginx.conf exists."""
        nginx_conf = PROJECT_ROOT / "deploy" / "config" / "nginx.conf"
        self.test(
            "nginx.conf created",
            nginx_conf.exists(),
            "Configuration file for NGINX reverse proxy"
        )
        
        if nginx_conf.exists():
            content = nginx_conf.read_text()
            has_upstream = "upstream ttrpg_backend" in content
            has_ssl = "ssl_protocols" in content
            has_rate_limit = "limit_req_zone" in content
            
            self.test(
                "nginx.conf properly configured",
                all([has_upstream, has_ssl, has_rate_limit]),
                "Has upstream, SSL, and rate limiting configurations"
            )
    
    def test_backup_delete_option(self):
        """Test backup manager --delete option."""
        backup_manager = PROJECT_ROOT / "deploy" / "backup" / "backup_manager.py"
        if not backup_manager.exists():
            self.test("backup_manager.py exists", False)
            return
        
        content = backup_manager.read_text()
        has_delete_method = "def delete_backup" in content
        has_delete_arg = "'--delete'" in content
        
        self.test(
            "backup_manager.py has delete functionality",
            has_delete_method and has_delete_arg,
            "delete_backup() method and --delete CLI argument implemented"
        )
    
    def test_backup_consistency(self):
        """Test backup consistency between systems."""
        # Both should use BackupManager
        backup_manager = PROJECT_ROOT / "deploy" / "backup" / "backup_manager.py"
        restore_manager = PROJECT_ROOT / "deploy" / "backup" / "restore_manager.py"
        
        if backup_manager.exists() and restore_manager.exists():
            restore_content = restore_manager.read_text()
            # Check if restore_manager can create pre-restore backups using BackupManager
            uses_backup_manager = "from deploy.backup.backup_manager import BackupManager" in restore_content
            
            self.test(
                "Backup systems are consistent",
                uses_backup_manager,
                "restore_manager uses BackupManager for pre-restore backups"
            )
    
    def test_makefile_improvements(self):
        """Test Makefile improvements."""
        makefile = PROJECT_ROOT / "Makefile"
        if not makefile.exists():
            self.test("Makefile exists", False)
            return
        
        content = makefile.read_text()
        
        # Check for version script usage
        uses_version_script = "deploy/scripts/get_version.py" in content
        
        # Check that inline Python is minimized
        inline_python_count = content.count('python -c "')
        
        self.test(
            "Makefile uses external version script",
            uses_version_script,
            "Using deploy/scripts/get_version.py instead of inline Python"
        )
        
        self.test(
            "Makefile minimizes inline Python",
            inline_python_count <= 2,  # Allow a couple for simple operations
            f"Only {inline_python_count} inline Python commands"
        )
    
    def test_docker_optimization(self):
        """Test Docker layer optimization."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        if not dockerfile.exists():
            self.test("Dockerfile exists", False)
            return
        
        content = dockerfile.read_text()
        lines = content.split('\n')
        
        # Check optimization patterns
        copy_requirements_first = False
        copy_source_later = False
        uses_no_cache = "--no-cache-dir" in content
        
        for i, line in enumerate(lines):
            if "COPY requirements.txt" in line:
                copy_requirements_first = True
                # Check if source copy comes later
                for j in range(i+1, len(lines)):
                    if "COPY . /build/" in lines[j]:
                        copy_source_later = True
                        break
        
        self.test(
            "Dockerfile optimized for layer caching",
            copy_requirements_first and copy_source_later,
            "Requirements copied before source for better caching"
        )
        
        self.test(
            "Dockerfile uses --no-cache-dir for pip",
            uses_no_cache,
            "Reduces image size by not caching pip downloads"
        )
    
    def test_version_extraction(self):
        """Test version extraction script."""
        get_version = PROJECT_ROOT / "deploy" / "scripts" / "get_version.py"
        self.test(
            "get_version.py script created",
            get_version.exists(),
            "Separate script for version extraction"
        )
        
        if get_version.exists():
            # Try running it
            try:
                result = subprocess.run(
                    [sys.executable, str(get_version)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version_output = result.stdout.strip()
                # Check if it outputs a version-like string
                is_version = len(version_output.split('.')) >= 2
                
                self.test(
                    "get_version.py works correctly",
                    result.returncode == 0 and is_version,
                    f"Outputs: {version_output}"
                )
            except Exception as e:
                self.test("get_version.py works correctly", False, str(e))
    
    def test_requirements_cleanup(self):
        """Test requirements-cpu.txt removal."""
        requirements_cpu = PROJECT_ROOT / "requirements-cpu.txt"
        self.test(
            "requirements-cpu.txt removed",
            not requirements_cpu.exists(),
            "Unused file has been cleaned up"
        )


def main():
    """Run all tests."""
    tester = CodeReviewTester()
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)
    
    print("\n✅ All code review fixes have been successfully implemented!")


if __name__ == "__main__":
    main()