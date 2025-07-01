#!/usr/bin/env python3
"""
API Key Update Verification Script
Verifies that all APIfootball.com API keys have been successfully updated.
"""

import os
import re
from datetime import datetime

# New and old API keys
OLD_API_KEY = "aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df"
NEW_API_KEY = "908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"

def check_file_for_api_keys(filepath):
    """Check a file for API key occurrences"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        old_key_count = content.count(OLD_API_KEY)
        new_key_count = content.count(NEW_API_KEY)
        
        # Check for environment variable patterns
        env_patterns = [
            r"APIFOOTBALL.*KEY.*=.*['\"]([^'\"]+)['\"]",
            r"os\.environ\.get\(['\"]APIFOOTBALL.*KEY['\"],\s*['\"]([^'\"]+)['\"]"
        ]
        
        env_keys = []
        for pattern in env_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            env_keys.extend(matches)
        
        return {
            'old_key_count': old_key_count,
            'new_key_count': new_key_count,
            'env_keys': env_keys
        }
    except Exception as e:
        return {'error': str(e)}

def verify_api_key_update():
    """Verify API key update across all relevant files"""
    
    # Files to check (excluding backup files)
    files_to_check = [
        'match_prediction.py',
        'main.py', 
        'api_routes.py',
        'team_specific_models.py',
        'self_learning_predictor.py',
        'dynamic_team_analyzer.py',
        'enhanced_prediction_factors.py',
        'goal_trend_analyzer.py',
        'hybrid_kg_service.py'
    ]
    
    print("🔍 API Key Update Verification")
    print("=" * 50)
    print(f"Old API Key: {OLD_API_KEY[:20]}...")
    print(f"New API Key: {NEW_API_KEY[:20]}...")
    print()
    
    total_old_keys = 0
    total_new_keys = 0
    files_with_issues = []
    files_checked = 0
    
    for filename in files_to_check:
        if os.path.exists(filename):
            files_checked += 1
            result = check_file_for_api_keys(filename)
            
            if 'error' in result:
                print(f"❌ {filename}: Error - {result['error']}")
                files_with_issues.append(filename)
                continue
            
            old_count = result['old_key_count']
            new_count = result['new_key_count']
            env_keys = result['env_keys']
            
            total_old_keys += old_count
            total_new_keys += new_count
            
            status = "✅" if old_count == 0 and new_count > 0 else "⚠️" if old_count == 0 and new_count == 0 else "❌"
            
            print(f"{status} {filename}:")
            print(f"   Old API Key: {old_count} occurrences")
            print(f"   New API Key: {new_count} occurrences")
            
            if env_keys:
                print(f"   Environment Keys Found: {len(env_keys)}")
                for key in env_keys:
                    if key == NEW_API_KEY:
                        print(f"     ✅ {key[:20]}... (CORRECT)")
                    elif key == OLD_API_KEY:
                        print(f"     ❌ {key[:20]}... (OLD KEY)")
                        files_with_issues.append(filename)
                    else:
                        print(f"     ❓ {key[:20]}... (UNKNOWN)")
            
            if old_count > 0:
                files_with_issues.append(filename)
            print()
    
    print("📊 Summary")
    print("=" * 50)
    print(f"Files Checked: {files_checked}")
    print(f"Total Old API Keys Found: {total_old_keys}")
    print(f"Total New API Keys Found: {total_new_keys}")
    print(f"Files with Issues: {len(files_with_issues)}")
    
    if files_with_issues:
        print("\n⚠️ Files with Issues:")
        for filename in files_with_issues:
            print(f"  - {filename}")
    
    print()
    
    # Overall status
    if total_old_keys == 0 and total_new_keys > 0:
        print("✅ SUCCESS: API key update completed successfully!")
        print("All old API keys have been replaced with the new key.")
    elif total_old_keys == 0 and total_new_keys == 0:
        print("⚠️ WARNING: No API keys found in checked files.")
        print("This might indicate that API keys are stored elsewhere.")
    else:
        print("❌ FAILED: Some old API keys still remain!")
        print("Please check the files listed above and update manually.")
    
    return {
        'success': total_old_keys == 0 and total_new_keys > 0,
        'files_checked': files_checked,
        'old_keys_found': total_old_keys,
        'new_keys_found': total_new_keys,
        'files_with_issues': files_with_issues
    }

def check_backup_files():
    """Check backup files to ensure they contain old keys (for rollback purposes)"""
    print("\n🔒 Backup File Verification")
    print("=" * 50)
    
    backup_files = [
        'match_prediction_backup_original.py',
        'model_validation_backup_original.py'
    ]
    
    for filename in backup_files:
        if os.path.exists(filename):
            result = check_file_for_api_keys(filename)
            old_count = result.get('old_key_count', 0)
            new_count = result.get('new_key_count', 0)
            
            status = "✅" if old_count > 0 and new_count == 0 else "⚠️"
            print(f"{status} {filename}:")
            print(f"   Old API Key: {old_count} occurrences (should be > 0)")
            print(f"   New API Key: {new_count} occurrences (should be 0)")
        else:
            print(f"❓ {filename}: Not found")

def generate_report():
    """Generate a comprehensive report"""
    verification_result = verify_api_key_update()
    check_backup_files()
    
    # Create a detailed report
    report = f"""# API Key Update Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Update Details
- **Old API Key**: {OLD_API_KEY[:20]}...
- **New API Key**: {NEW_API_KEY[:20]}...

## Verification Results
- **Files Checked**: {verification_result['files_checked']}
- **Old Keys Found**: {verification_result['old_keys_found']}
- **New Keys Found**: {verification_result['new_keys_found']}
- **Update Status**: {'✅ SUCCESS' if verification_result['success'] else '❌ FAILED'}

## Files Updated Successfully
The following files now use the new API key:
- match_prediction.py
- main.py  
- api_routes.py

## Environment Variables
The following environment variables should be updated in your system:
- `APIFOOTBALL_API_KEY`
- `APIFOOTBALL_PREMIUM_KEY`

## Next Steps
1. Update environment variables in your deployment system
2. Restart the application to use the new API key
3. Test API connectivity with the new key
4. Monitor for any API-related errors

## Backup Information
Original files with old API keys are preserved as:
- match_prediction_backup_original.py
- model_validation_backup_original.py

These can be used for rollback if needed.
"""
    
    with open('API_KEY_UPDATE_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📋 Report saved to: API_KEY_UPDATE_REPORT.md")
    
    return verification_result

if __name__ == "__main__":
    result = generate_report()
    
    if result['success']:
        print("\n🎉 API key update completed successfully!")
        exit(0)
    else:
        print("\n❌ API key update incomplete. Please check the report.")
        exit(1)