"""
多語言本地化系統 - Internationalization (i18n) Support
支援中文和英文的界面文本、錯誤信息和API響應
"""

from typing import Dict, Any, Optional
import json
import os

class LocalizationManager:
    """本地化管理器"""
    
    def __init__(self, default_language: str = 'zh'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = {}
        self._load_translations()
    
    def _load_translations(self):
        """載入翻譯文件"""
        self.translations = {
            'zh': {
                # 通用
                'app_title': '詐騙檢測API - 客戶門戶',
                'app_subtitle': '管理您的API密鑰、監控使用量、查看計費信息',
                'login': '登入',
                'logout': '登出',
                'register': '註冊',
                'email': '電子郵件',
                'password': '密碼',
                'company_name': '公司名稱',
                'save': '保存',
                'cancel': '取消',
                'delete': '刪除',
                'edit': '編輯',
                'create': '創建',
                'update': '更新',
                'success': '成功',
                'error': '錯誤',
                'warning': '警告',
                'info': '信息',
                'loading': '載入中...',
                'no_data': '暫無數據',
                
                # 導航
                'nav_dashboard': '儀表板',
                'nav_api_keys': 'API密鑰管理',
                'nav_usage': '使用量監控',
                'nav_billing': '計費信息',
                'nav_sla': 'SLA狀態',
                'nav_settings': '帳戶設定',
                
                # 儀表板
                'dashboard_title': '客戶儀表板',
                'monthly_cost': '本月費用',
                'monthly_requests': '本月請求',
                'avg_response_time': '平均響應時間',
                'api_keys_count': 'API密鑰',
                'usage_trend': '使用量趨勢',
                'performance_metrics': '性能指標',
                'active_alerts': '活躍警報',
                'no_alerts': '目前沒有活躍警報',
                'estimated_monthly_cost': '預估月度成本',
                
                # API密鑰管理
                'api_keys_title': 'API密鑰管理',
                'create_api_key': '創建新API密鑰',
                'key_name': '密鑰名稱',
                'permissions': '權限',
                'expires_in_days': '有效期（天）',
                'allowed_ips': '允許的IP地址（每行一個）',
                'api_key_created': 'API密鑰創建成功！',
                'api_key_warning': '⚠️ 請妥善保管您的API密鑰，我們不會再次顯示完整密鑰',
                'existing_keys': '現有API密鑰',
                'key_status': '狀態',
                'usage_count': '使用次數',
                'last_used': '最後使用',
                'never_used': '從未使用',
                'revoke_key': '撤銷密鑰',
                'key_details': '詳細信息',
                'basic_info': '基本信息',
                'created_date': '創建日期',
                'expiry_date': '到期日期',
                'rate_limits': '速率限制',
                'per_minute': '每分鐘',
                'per_hour': '每小時',
                'per_day': '每天',
                
                # 使用量監控
                'usage_monitoring_title': '使用量監控',
                'time_range': '時間範圍',
                'last_24_hours': '最近24小時',
                'last_7_days': '最近7天',
                'last_30_days': '最近30天',
                'custom': '自定義',
                'total_requests': '總請求數',
                'success_rate': '成功率',
                'http_status_distribution': 'HTTP狀態碼分佈',
                'request_status_distribution': '請求狀態分佈',
                'realtime_metrics': '實時指標',
                'response_time_stats': '響應時間統計 (最近1小時)',
                'average': '平均',
                'minimum': '最小',
                'maximum': '最大',
                'throughput_stats': '吞吐量統計',
                'requests_per_minute': '每分鐘請求數',
                
                # 計費信息
                'billing_title': '計費信息',
                'current_billing_period': '當前計費期間',
                'start_date': '開始日期',
                'end_date': '結束日期',
                'last_updated': '上次更新',
                'cost_summary': '費用摘要',
                'current_cost': '當前費用',
                'plan_difference': '與計劃差額',
                'usage_breakdown': '使用量分解',
                'service_type': '服務類型',
                'usage': '使用量',
                'unit_cost': '單位成本',
                'total_cost': '總成本',
                'cost_distribution': '成本分佈',
                'subscription_plans': '訂閱方案',
                'current_plan': '當前方案',
                'monthly_fee': '月費',
                'included': '包含',
                'predictions': '次預測',
                'upgrade_to': '升級到',
                'upgrade_feature_developing': '升級功能開發中...',
                
                # SLA狀態
                'sla_title': '服務等級協議 (SLA)',
                'all_services_normal': '所有服務運行正常',
                'service_issues': '服務存在問題',
                'sla_metrics': 'SLA指標',
                'uptime_24h': '24小時正常運行時間',
                'uptime_30d': '30天正常運行時間',
                'sla_commitments': 'SLA承諾',
                'target_value': '目標值',
                'warning_threshold': '警告閾值',
                'breach_threshold': '違約閾值',
                'measurement_period': '測量期間',
                'current_health_check': '當前健康檢查',
                'normal': '正常',
                'abnormal': '異常',
                
                # 帳戶設定
                'account_settings_title': '帳戶設定',
                'basic_information': '基本信息',
                'subscription_plan': '訂閱方案',
                'update_info': '更新信息',
                'account_updated': '帳戶信息已更新',
                'security_settings': '安全設定',
                'change_password': '變更密碼',
                'current_password': '當前密碼',
                'new_password': '新密碼',
                'confirm_password': '確認新密碼',
                'update_password': '更新密碼',
                'password_updated': '密碼已更新',
                'password_mismatch': '新密碼與確認密碼不符',
                'notification_settings': '通知設定',
                'usage_alerts': '使用量警報',
                'billing_notifications': '計費通知',
                'security_alerts': '安全警報',
                'maintenance_notifications': '維護通知',
                'feature_announcements': '新功能公告',
                'marketing_info': '營銷信息',
                'danger_zone': '危險區域',
                'delete_account': '刪除帳戶',
                'delete_account_warning': '**警告**: 刪除帳戶將永久移除所有數據，此操作無法撤銷。',
                'delete_confirmation': '輸入 \'DELETE\' 確認刪除帳戶',
                'permanently_delete': '永久刪除帳戶',
                'delete_feature_developing': '帳戶刪除功能開發中...',
                'confirm_delete_text': '請輸入 \'DELETE\' 確認刪除',
                
                # 登入頁面
                'customer_login': '客戶登入',
                'login_failed': '登入失敗，請檢查您的認證信息',
                'registration_success': '註冊成功！請登入',
                'registration_failed': '註冊失敗，該電子郵件可能已被使用',
                'fill_all_fields': '請填寫所有欄位',
                'demo_accounts': '演示帳號',
                'use_demo_account': '使用演示帳號登入',
                
                # 錯誤信息
                'invalid_api_key': '無效的API密鑰',
                'insufficient_permissions': '權限不足',
                'rate_limit_exceeded': '速率限制超出',
                'service_error': '服務錯誤',
                'prediction_error': '預測服務錯誤',
                'batch_prediction_error': '批量預測服務錯誤',
                'create_api_key_failed': '創建API密鑰失敗',
                'batch_size_limit': '批量大小不能超過10,000筆交易',
                
                # 單位和數值
                'currency_symbol': '$',
                'percentage': '%',
                'milliseconds': 'ms',
                'requests': '次請求',
                'predictions_unit': '次',
                'days': '天',
                'hours': '小時',
                'minutes': '分鐘',
                'seconds': '秒',
            },
            
            'en': {
                # Common
                'app_title': 'Fraud Detection API - Customer Portal',
                'app_subtitle': 'Manage your API keys, monitor usage, and view billing information',
                'login': 'Login',
                'logout': 'Logout',
                'register': 'Register',
                'email': 'Email',
                'password': 'Password',
                'company_name': 'Company Name',
                'save': 'Save',
                'cancel': 'Cancel',
                'delete': 'Delete',
                'edit': 'Edit',
                'create': 'Create',
                'update': 'Update',
                'success': 'Success',
                'error': 'Error',
                'warning': 'Warning',
                'info': 'Info',
                'loading': 'Loading...',
                'no_data': 'No data available',
                
                # Navigation
                'nav_dashboard': 'Dashboard',
                'nav_api_keys': 'API Key Management',
                'nav_usage': 'Usage Monitoring',
                'nav_billing': 'Billing Information',
                'nav_sla': 'SLA Status',
                'nav_settings': 'Account Settings',
                
                # Dashboard
                'dashboard_title': 'Customer Dashboard',
                'monthly_cost': 'Monthly Cost',
                'monthly_requests': 'Monthly Requests',
                'avg_response_time': 'Avg Response Time',
                'api_keys_count': 'API Keys',
                'usage_trend': 'Usage Trend',
                'performance_metrics': 'Performance Metrics',
                'active_alerts': 'Active Alerts',
                'no_alerts': 'No active alerts currently',
                'estimated_monthly_cost': 'Estimated Monthly Cost',
                
                # API Key Management
                'api_keys_title': 'API Key Management',
                'create_api_key': 'Create New API Key',
                'key_name': 'Key Name',
                'permissions': 'Permissions',
                'expires_in_days': 'Expires in (days)',
                'allowed_ips': 'Allowed IP Addresses (one per line)',
                'api_key_created': 'API key created successfully!',
                'api_key_warning': '⚠️ Please keep your API key secure. We will not show the complete key again',
                'existing_keys': 'Existing API Keys',
                'key_status': 'Status',
                'usage_count': 'Usage Count',
                'last_used': 'Last Used',
                'never_used': 'Never Used',
                'revoke_key': 'Revoke Key',
                'key_details': 'Details',
                'basic_info': 'Basic Information',
                'created_date': 'Created Date',
                'expiry_date': 'Expiry Date',
                'rate_limits': 'Rate Limits',
                'per_minute': 'Per Minute',
                'per_hour': 'Per Hour',
                'per_day': 'Per Day',
                
                # Usage Monitoring
                'usage_monitoring_title': 'Usage Monitoring',
                'time_range': 'Time Range',
                'last_24_hours': 'Last 24 Hours',
                'last_7_days': 'Last 7 Days',
                'last_30_days': 'Last 30 Days',
                'custom': 'Custom',
                'total_requests': 'Total Requests',
                'success_rate': 'Success Rate',
                'http_status_distribution': 'HTTP Status Code Distribution',
                'request_status_distribution': 'Request Status Distribution',
                'realtime_metrics': 'Real-time Metrics',
                'response_time_stats': 'Response Time Statistics (Last 1 Hour)',
                'average': 'Average',
                'minimum': 'Minimum',
                'maximum': 'Maximum',
                'throughput_stats': 'Throughput Statistics',
                'requests_per_minute': 'Requests Per Minute',
                
                # Billing Information
                'billing_title': 'Billing Information',
                'current_billing_period': 'Current Billing Period',
                'start_date': 'Start Date',
                'end_date': 'End Date',
                'last_updated': 'Last Updated',
                'cost_summary': 'Cost Summary',
                'current_cost': 'Current Cost',
                'plan_difference': 'Plan Difference',
                'usage_breakdown': 'Usage Breakdown',
                'service_type': 'Service Type',
                'usage': 'Usage',
                'unit_cost': 'Unit Cost',
                'total_cost': 'Total Cost',
                'cost_distribution': 'Cost Distribution',
                'subscription_plans': 'Subscription Plans',
                'current_plan': 'Current Plan',
                'monthly_fee': 'Monthly Fee',
                'included': 'Included',
                'predictions': ' predictions',
                'upgrade_to': 'Upgrade to',
                'upgrade_feature_developing': 'Upgrade feature in development...',
                
                # SLA Status
                'sla_title': 'Service Level Agreement (SLA)',
                'all_services_normal': 'All services running normally',
                'service_issues': 'Service issues detected',
                'sla_metrics': 'SLA Metrics',
                'uptime_24h': '24-Hour Uptime',
                'uptime_30d': '30-Day Uptime',
                'sla_commitments': 'SLA Commitments',
                'target_value': 'Target Value',
                'warning_threshold': 'Warning Threshold',
                'breach_threshold': 'Breach Threshold',
                'measurement_period': 'Measurement Period',
                'current_health_check': 'Current Health Check',
                'normal': 'Normal',
                'abnormal': 'Abnormal',
                
                # Account Settings
                'account_settings_title': 'Account Settings',
                'basic_information': 'Basic Information',
                'subscription_plan': 'Subscription Plan',
                'update_info': 'Update Information',
                'account_updated': 'Account information updated',
                'security_settings': 'Security Settings',
                'change_password': 'Change Password',
                'current_password': 'Current Password',
                'new_password': 'New Password',
                'confirm_password': 'Confirm New Password',
                'update_password': 'Update Password',
                'password_updated': 'Password updated',
                'password_mismatch': 'New password and confirmation do not match',
                'notification_settings': 'Notification Settings',
                'usage_alerts': 'Usage Alerts',
                'billing_notifications': 'Billing Notifications',
                'security_alerts': 'Security Alerts',
                'maintenance_notifications': 'Maintenance Notifications',
                'feature_announcements': 'Feature Announcements',
                'marketing_info': 'Marketing Information',
                'danger_zone': 'Danger Zone',
                'delete_account': 'Delete Account',
                'delete_account_warning': '**Warning**: Deleting your account will permanently remove all data. This action cannot be undone.',
                'delete_confirmation': 'Type \'DELETE\' to confirm account deletion',
                'permanently_delete': 'Permanently Delete Account',
                'delete_feature_developing': 'Account deletion feature in development...',
                'confirm_delete_text': 'Please type \'DELETE\' to confirm deletion',
                
                # Login Page
                'customer_login': 'Customer Login',
                'login_failed': 'Login failed. Please check your credentials',
                'registration_success': 'Registration successful! Please login',
                'registration_failed': 'Registration failed. Email may already be in use',
                'fill_all_fields': 'Please fill in all fields',
                'demo_accounts': 'Demo Account',
                'use_demo_account': 'Use Demo Account Login',
                
                # Error Messages
                'invalid_api_key': 'Invalid API key',
                'insufficient_permissions': 'Insufficient permissions',
                'rate_limit_exceeded': 'Rate limit exceeded',
                'service_error': 'Service error',
                'prediction_error': 'Prediction service error',
                'batch_prediction_error': 'Batch prediction service error',
                'create_api_key_failed': 'Failed to create API key',
                'batch_size_limit': 'Batch size cannot exceed 10,000 transactions',
                
                # Units and Values
                'currency_symbol': '$',
                'percentage': '%',
                'milliseconds': 'ms',
                'requests': ' requests',
                'predictions_unit': ' calls',
                'days': ' days',
                'hours': ' hours',
                'minutes': ' minutes',
                'seconds': ' seconds',
            }
        }
    
    def set_language(self, language: str):
        """設置當前語言"""
        if language in self.translations:
            self.current_language = language
        else:
            self.current_language = self.default_language
    
    def get_text(self, key: str, **kwargs) -> str:
        """獲取翻譯文本"""
        text = self.translations.get(self.current_language, {}).get(key)
        
        if text is None:
            # 回退到默認語言
            text = self.translations.get(self.default_language, {}).get(key, key)
        
        # 處理格式化參數
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError):
                pass
        
        return text
    
    def get_current_language(self) -> str:
        """獲取當前語言"""
        return self.current_language
    
    def get_available_languages(self) -> list:
        """獲取可用語言列表"""
        return list(self.translations.keys())
    
    def translate_dict(self, data: Dict[str, Any], key_mappings: Dict[str, str]) -> Dict[str, Any]:
        """翻譯字典中的特定鍵值"""
        translated = data.copy()
        
        for original_key, translation_key in key_mappings.items():
            if original_key in translated:
                translated[original_key] = self.get_text(translation_key)
        
        return translated

# 全局本地化管理器實例
_localization_manager = LocalizationManager()

def get_localization_manager() -> LocalizationManager:
    """獲取本地化管理器實例"""
    return _localization_manager

def set_language(language: str):
    """設置全局語言"""
    _localization_manager.set_language(language)

def t(key: str, **kwargs) -> str:
    """便捷的翻譯函數"""
    return _localization_manager.get_text(key, **kwargs)

def get_language() -> str:
    """獲取當前語言"""
    return _localization_manager.get_current_language()

# 訂閱方案翻譯
PLAN_TRANSLATIONS = {
    'zh': {
        'free': {
            'name': '免費方案',
            'description': '適合個人開發者和小型項目'
        },
        'basic': {
            'name': '基礎方案',
            'description': '適合中小型企業'
        },
        'professional': {
            'name': '專業方案',
            'description': '適合大型企業'
        },
        'enterprise': {
            'name': '企業方案',
            'description': '適合大型企業和金融機構'
        }
    },
    'en': {
        'free': {
            'name': 'Free Plan',
            'description': 'Perfect for individual developers and small projects'
        },
        'basic': {
            'name': 'Basic Plan',
            'description': 'Suitable for small to medium businesses'
        },
        'professional': {
            'name': 'Professional Plan',
            'description': 'Designed for large enterprises'
        },
        'enterprise': {
            'name': 'Enterprise Plan',
            'description': 'For large enterprises and financial institutions'
        }
    }
}

def get_plan_translation(plan_type: str, field: str = 'name') -> str:
    """獲取方案翻譯"""
    language = get_language()
    return PLAN_TRANSLATIONS.get(language, {}).get(plan_type, {}).get(field, plan_type)

# 錯誤信息翻譯
ERROR_TRANSLATIONS = {
    'zh': {
        'invalid_api_key': '無效的API密鑰',
        'insufficient_permissions': '權限不足',
        'rate_limit_exceeded': '請求頻率超出限制',
        'service_unavailable': '服務暫時不可用',
        'internal_server_error': '內部服務器錯誤',
        'bad_request': '請求格式錯誤',
        'unauthorized': '未授權訪問',
        'forbidden': '禁止訪問',
        'not_found': '資源不存在',
        'method_not_allowed': '方法不被允許',
        'request_timeout': '請求超時',
        'too_many_requests': '請求過於頻繁',
    },
    'en': {
        'invalid_api_key': 'Invalid API key',
        'insufficient_permissions': 'Insufficient permissions',
        'rate_limit_exceeded': 'Rate limit exceeded',
        'service_unavailable': 'Service temporarily unavailable',
        'internal_server_error': 'Internal server error',
        'bad_request': 'Bad request format',
        'unauthorized': 'Unauthorized access',
        'forbidden': 'Access forbidden',
        'not_found': 'Resource not found',
        'method_not_allowed': 'Method not allowed',
        'request_timeout': 'Request timeout',
        'too_many_requests': 'Too many requests',
    }
}

def get_error_message(error_code: str) -> str:
    """獲取錯誤信息翻譯"""
    language = get_language()
    return ERROR_TRANSLATIONS.get(language, {}).get(error_code, error_code)

if __name__ == "__main__":
    # 測試本地化功能
    lm = LocalizationManager()
    
    print("測試中文:")
    lm.set_language('zh')
    print(f"應用標題: {lm.get_text('app_title')}")
    print(f"登入: {lm.get_text('login')}")
    
    print("\n測試英文:")
    lm.set_language('en')
    print(f"App Title: {lm.get_text('app_title')}")
    print(f"Login: {lm.get_text('login')}")
    
    print("\n便捷函數測試:")
    set_language('zh')
    print(f"當前語言: {get_language()}")
    print(f"儀表板: {t('nav_dashboard')}")
    
    set_language('en')
    print(f"Current Language: {get_language()}")
    print(f"Dashboard: {t('nav_dashboard')}")