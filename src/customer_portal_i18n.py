"""
多語言客戶門戶和管理面板 - International Customer Portal
支援中文和英文的客戶自助服務門戶和管理面板功能
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import hashlib
import uuid

from .billing_system import BillingEngine, PlanType, create_billing_engine
from .api_key_management import APIKeyManager, Permission, get_default_rate_limits, create_api_key_manager
from .usage_monitoring import UsageMonitoringSystem, create_usage_monitoring_system
from .sla_management import SLAManager, create_sla_manager
from .localization import get_localization_manager, t, set_language, get_language, get_plan_translation

class InternationalCustomerPortal:
    """國際化客戶門戶類"""
    
    def __init__(self):
        self.billing_engine = create_billing_engine()
        self.api_key_manager = create_api_key_manager()
        self.sla_manager = create_sla_manager()
        self.usage_monitoring = create_usage_monitoring_system(
            self.billing_engine, self.api_key_manager
        )
        self.lm = get_localization_manager()
        
        # 初始化session state
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'customer_id' not in st.session_state:
            st.session_state.customer_id = None
        if 'customer_info' not in st.session_state:
            st.session_state.customer_info = None
        if 'language' not in st.session_state:
            st.session_state.language = 'zh'
        
        # 設置語言
        set_language(st.session_state.language)
    
    def run(self):
        """運行國際化客戶門戶應用"""
        st.set_page_config(
            page_title=t('app_title'),
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 語言選擇器（頂部）
        self._show_language_selector()
        
        # 自定義CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
        .language-selector {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background: white;
            padding: 5px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 主標題
        st.markdown(f"""
        <div class="main-header">
            <h1>🛡️ {t('app_title')}</h1>
            <p>{t('app_subtitle')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 檢查登入狀態
        if not st.session_state.logged_in:
            self._show_login_page()
        else:
            self._show_main_portal()
    
    def _show_language_selector(self):
        """顯示語言選擇器"""
        col1, col2, col3 = st.columns([4, 1, 1])
        
        with col2:
            language_options = {
                'zh': '中文',
                'en': 'English'
            }
            
            selected_language = st.selectbox(
                "Language / 語言",
                options=list(language_options.keys()),
                format_func=lambda x: language_options[x],
                index=0 if st.session_state.language == 'zh' else 1,
                key="language_selector"
            )
            
            if selected_language != st.session_state.language:
                st.session_state.language = selected_language
                set_language(selected_language)
                st.rerun()
    
    def _show_login_page(self):
        """顯示登入頁面"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"### 🔐 {t('customer_login')}")
            
            with st.form("login_form"):
                email = st.text_input(f"📧 {t('email')}")
                password = st.text_input(f"🔒 {t('password')}", type="password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_button = st.form_submit_button(t('login'), use_container_width=True)
                with col_b:
                    register_button = st.form_submit_button(t('register'), use_container_width=True)
                
                if login_button:
                    if self._authenticate_customer(email, password):
                        st.success(t('success'))
                        st.rerun()
                    else:
                        st.error(t('login_failed'))
                
                if register_button:
                    if email and password:
                        customer_id = self._register_customer(email, password)
                        if customer_id:
                            st.success(t('registration_success'))
                        else:
                            st.error(t('registration_failed'))
                    else:
                        st.error(t('fill_all_fields'))
            
            # 演示帳號
            st.markdown("---")
            st.markdown(f"### 🎯 {t('demo_accounts')}")
            if st.button(t('use_demo_account'), use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.customer_id = "demo_customer"
                st.session_state.customer_info = {
                    'email': 'demo@example.com',
                    'company_name': 'Demo Company',
                    'plan_type': 'professional'
                }
                st.rerun()
    
    def _show_main_portal(self):
        """顯示主門戶"""
        # 側邊欄
        with st.sidebar:
            st.markdown(f"### 👤 {t('basic_information')}")
            customer_info = st.session_state.customer_info
            st.write(f"**{t('company_name')}**: {customer_info.get('company_name', 'N/A')}")
            plan_name = get_plan_translation(customer_info.get('plan_type', 'free'))
            st.write(f"**{t('subscription_plan')}**: {plan_name}")
            st.write(f"**ID**: {st.session_state.customer_id}")
            
            st.markdown("---")
            
            # 導航菜單
            page = st.selectbox(
                f"📋 {t('nav_dashboard')}",
                [
                    t('nav_dashboard'),
                    t('nav_api_keys'), 
                    t('nav_usage'),
                    t('nav_billing'),
                    t('nav_sla'),
                    t('nav_settings')
                ]
            )
            
            st.markdown("---")
            
            if st.button(t('logout'), use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.customer_id = None
                st.session_state.customer_info = None
                st.rerun()
        
        # 主要內容區域
        if page == t('nav_dashboard'):
            self._show_dashboard()
        elif page == t('nav_api_keys'):
            self._show_api_key_management()
        elif page == t('nav_usage'):
            self._show_usage_monitoring()
        elif page == t('nav_billing'):
            self._show_billing_info()
        elif page == t('nav_sla'):
            self._show_sla_status()
        elif page == t('nav_settings'):
            self._show_account_settings()
    
    def _show_dashboard(self):
        """顯示儀表板"""
        st.markdown(f"## 📊 {t('dashboard_title')}")
        
        customer_id = st.session_state.customer_id
        
        # 獲取儀表板數據
        dashboard_data = self.usage_monitoring.get_customer_usage_dashboard(customer_id)
        real_time_bill = dashboard_data.get('real_time_bill')
        metrics = dashboard_data.get('metrics', {})
        alerts = dashboard_data.get('alerts', [])
        
        # 頂部指標卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_cost = real_time_bill.get('estimated_monthly_cost', 0) if real_time_bill else 0
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 {t('monthly_cost')}</h4>
                <h2>{t('currency_symbol')}{monthly_cost:.2f}</h2>
                <p>{t('estimated_monthly_cost')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_requests = sum(real_time_bill.get('usage_summary', {}).values()) if real_time_bill else 0
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 {t('monthly_requests')}</h4>
                <h2>{total_requests:,}</h2>
                <p>API {t('requests')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_response_time = metrics.get('response_time', {}).get('average', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚡ {t('avg_response_time')}</h4>
                <h2>{avg_response_time:.0f}{t('milliseconds')}</h2>
                <p>{t('last_24_hours')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            api_keys_count = dashboard_data.get('api_keys_count', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔑 {t('api_keys_count')}</h4>
                <h2>{api_keys_count}</h2>
                <p>{t('create')} API Key</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 圖表區域
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 📊 {t('usage_trend')}")
            if real_time_bill and real_time_bill.get('usage_summary'):
                usage_data = real_time_bill['usage_summary']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(usage_data.keys()),
                        y=list(usage_data.values()),
                        marker_color='#667eea'
                    )
                ])
                fig.update_layout(
                    title=t('usage_trend'),
                    xaxis_title=t('service_type'),
                    yaxis_title=t('requests'),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(t('no_data'))
        
        with col2:
            st.markdown(f"### ⚡ {t('performance_metrics')}")
            if metrics.get('response_time'):
                response_metrics = metrics['response_time']
                
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=response_metrics.get('average', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{t('avg_response_time')} ({t('milliseconds')})"},
                    delta={'reference': 500},
                    gauge={
                        'axis': {'range': [None, 1000]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 500], 'color': "lightgray"},
                            {'range': [500, 750], 'color': "yellow"},
                            {'range': [750, 1000], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 500
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(t('no_data'))
        
        # 警報區域
        if alerts:
            st.markdown(f"### 🚨 {t('active_alerts')}")
            for alert in alerts:
                alert_type = alert.get('alert_type', t('error'))
                message = alert.get('message', t('no_data'))
                
                if 'high' in alert_type.lower():
                    st.error(f"**{alert_type}**: {message}")
                elif 'medium' in alert_type.lower():
                    st.warning(f"**{alert_type}**: {message}")
                else:
                    st.info(f"**{alert_type}**: {message}")
        else:
            st.success(f"🟢 {t('no_alerts')}")
    
    def _show_api_key_management(self):
        """顯示API密鑰管理"""
        st.markdown(f"## 🔑 {t('api_keys_title')}")
        
        customer_id = st.session_state.customer_id
        
        # 獲取現有密鑰
        api_keys = self.api_key_manager.get_customer_keys(customer_id)
        
        # 創建新密鑰區域
        with st.expander(f"➕ {t('create_api_key')}", expanded=False):
            with st.form("create_api_key"):
                col1, col2 = st.columns(2)
                
                with col1:
                    key_name = st.text_input(t('key_name'), placeholder="Production API Key")
                    permissions = st.multiselect(
                        t('permissions'),
                        options=[p.value for p in Permission],
                        default=['predict', 'read']
                    )
                
                with col2:
                    expires_in_days = st.number_input(t('expires_in_days'), min_value=1, max_value=3650, value=365)
                    allowed_ips = st.text_area(
                        t('allowed_ips'),
                        placeholder="192.168.1.1\n10.0.0.1",
                        height=100
                    )
                
                if st.form_submit_button(t('create_api_key'), use_container_width=True):
                    if key_name:
                        try:
                            # 轉換權限
                            perm_objects = [Permission(p) for p in permissions]
                            
                            # 處理IP列表
                            ip_list = [ip.strip() for ip in allowed_ips.split('\n') if ip.strip()] if allowed_ips else []
                            
                            # 獲取客戶計劃的默認速率限制
                            plan_type = st.session_state.customer_info.get('plan_type', 'free')
                            rate_limits = get_default_rate_limits(plan_type)
                            
                            # 創建密鑰
                            raw_key, api_key = self.api_key_manager.create_api_key(
                                customer_id=customer_id,
                                name=key_name,
                                permissions=perm_objects,
                                rate_limits=rate_limits,
                                expires_in_days=expires_in_days,
                                allowed_ips=ip_list
                            )
                            
                            st.success(t('api_key_created'))
                            st.code(f"API Key: {raw_key}")
                            st.warning(t('api_key_warning'))
                            
                            # 刷新頁面
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"{t('create_api_key_failed')}: {str(e)}")
                    else:
                        st.error(t('fill_all_fields'))
        
        # 現有密鑰列表
        st.markdown(f"### 📋 {t('existing_keys')}")
        
        if api_keys:
            for api_key in api_keys:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        status_color = "🟢" if api_key.is_valid() else "🔴"
                        st.write(f"{status_color} **{api_key.name}**")
                        st.caption(f"ID: {api_key.id[:8]}...")
                    
                    with col2:
                        st.write(f"**{t('permissions')}**: {len(api_key.permissions)}")
                        permissions_str = ", ".join([p.value for p in api_key.permissions])
                        st.caption(permissions_str)
                    
                    with col3:
                        st.write(f"**{t('usage_count')}**: {api_key.usage_count:,}")
                        if api_key.last_used_at:
                            st.caption(f"{t('last_used')}: {api_key.last_used_at.strftime('%Y-%m-%d')}")
                        else:
                            st.caption(t('never_used'))
                    
                    with col4:
                        if st.button("🗑️", key=f"delete_{api_key.id}", help=t('revoke_key')):
                            if self.api_key_manager.revoke_api_key(api_key.id):
                                st.success(t('success'))
                                st.rerun()
                            else:
                                st.error(t('error'))
                    
                    # 詳細信息
                    with st.expander(f"📊 {api_key.name} - {t('key_details')}"):
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            st.write(f"**{t('basic_info')}**")
                            st.write(f"{t('created_date')}: {api_key.created_at.strftime('%Y-%m-%d %H:%M')}")
                            if api_key.expires_at:
                                st.write(f"{t('expiry_date')}: {api_key.expires_at.strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"{t('key_status')}: {api_key.status.value}")
                        
                        with info_col2:
                            st.write(f"**{t('rate_limits')}**")
                            st.write(f"{t('per_minute')}: {api_key.rate_limits.requests_per_minute}")
                            st.write(f"{t('per_hour')}: {api_key.rate_limits.requests_per_hour}")
                            st.write(f"{t('per_day')}: {api_key.rate_limits.requests_per_day}")
                        
                        if api_key.allowed_ips:
                            st.write(f"**{t('allowed_ips')}**")
                            for ip in api_key.allowed_ips:
                                st.code(ip)
                    
                    st.markdown("---")
        else:
            st.info(t('no_data'))
    
    def _show_usage_monitoring(self):
        """顯示使用量監控"""
        st.markdown(f"## 📈 {t('usage_monitoring_title')}")
        
        customer_id = st.session_state.customer_id
        
        # 時間範圍選擇
        col1, col2 = st.columns(2)
        with col1:
            time_range = st.selectbox(
                t('time_range'),
                [t('last_24_hours'), t('last_7_days'), t('last_30_days'), t('custom')]
            )
        
        with col2:
            if time_range == t('custom'):
                date_range = st.date_input(
                    t('custom'),
                    value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
                    max_value=datetime.now().date()
                )
        
        # 獲取使用統計
        if time_range == t('last_24_hours'):
            days = 1
        elif time_range == t('last_7_days'):
            days = 7
        elif time_range == t('last_30_days'):
            days = 30
        else:
            days = 7  # 默認
        
        usage_stats = self.api_key_manager.get_usage_statistics(customer_id, days)
        
        # 顯示統計數據
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                t('total_requests'),
                f"{usage_stats.get('total_requests', 0):,}",
                delta=None
            )
        
        with col2:
            success_rate = usage_stats.get('success_rate', 0)
            st.metric(
                t('success_rate'),
                f"{success_rate:.1f}{t('percentage')}",
                delta=f"{success_rate - 95:.1f}{t('percentage')}" if success_rate > 0 else None
            )
        
        with col3:
            avg_response_time = usage_stats.get('average_response_time_ms', 0)
            st.metric(
                t('avg_response_time'),
                f"{avg_response_time:.0f}{t('milliseconds')}",
                delta=None
            )
        
        # 狀態碼分佈圖
        if usage_stats.get('status_distribution'):
            st.markdown(f"### 📊 {t('http_status_distribution')}")
            
            status_data = usage_stats['status_distribution']
            
            fig = px.pie(
                values=list(status_data.values()),
                names=[f"HTTP {code}" for code in status_data.keys()],
                title=t('request_status_distribution')
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # 實時指標
        st.markdown(f"### ⚡ {t('realtime_metrics')}")
        
        dashboard_data = self.usage_monitoring.get_customer_usage_dashboard(customer_id)
        metrics = dashboard_data.get('metrics', {})
        
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'response_time' in metrics:
                    rt_metrics = metrics['response_time']
                    st.write(f"**{t('response_time_stats')}**")
                    st.write(f"{t('average')}: {rt_metrics.get('average', 0):.0f}{t('milliseconds')}")
                    st.write(f"{t('minimum')}: {rt_metrics.get('min', 0):.0f}{t('milliseconds')}")
                    st.write(f"{t('maximum')}: {rt_metrics.get('max', 0):.0f}{t('milliseconds')}")
                    st.write(f"P95: {rt_metrics.get('p95', 0):.0f}{t('milliseconds')}")
            
            with col2:
                if 'throughput' in metrics:
                    tp_metrics = metrics['throughput']
                    st.write(f"**{t('throughput_stats')}**")
                    st.write(f"{t('requests_per_minute')}: {tp_metrics.get('requests_per_minute', 0):.1f}")
                    st.write(f"{t('total_requests')}: {tp_metrics.get('count', 0)}")
        else:
            st.info(t('no_data'))
    
    def _show_billing_info(self):
        """顯示計費信息"""
        st.markdown(f"## 💰 {t('billing_title')}")
        
        customer_id = st.session_state.customer_id
        
        # 獲取實時帳單
        dashboard_data = self.usage_monitoring.get_customer_usage_dashboard(customer_id)
        real_time_bill = dashboard_data.get('real_time_bill')
        
        if real_time_bill:
            # 當前計費期間
            st.markdown(f"### 📅 {t('current_billing_period')}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{t('start_date')}**: {real_time_bill['current_period_start'][:10]}")
                st.write(f"**{t('end_date')}**: {real_time_bill['current_period_end'][:10]}")
            
            with col2:
                st.write(f"**{t('last_updated')}**: {real_time_bill['last_updated'][:19]}")
            
            # 費用摘要
            st.markdown(f"### 💸 {t('cost_summary')}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    t('current_cost'),
                    f"{t('currency_symbol')}{real_time_bill['total_cost']:.2f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    t('estimated_monthly_cost'),
                    f"{t('currency_symbol')}{real_time_bill['estimated_monthly_cost']:.2f}",
                    delta=None
                )
            
            with col3:
                # 計算節省/超支
                plan_limits = {
                    'free': 0,
                    'basic': 99,
                    'professional': 299,
                    'enterprise': 999
                }
                plan_type = st.session_state.customer_info.get('plan_type', 'free')
                plan_limit = plan_limits.get(plan_type, 0)
                
                if real_time_bill['estimated_monthly_cost'] > plan_limit:
                    delta_text = f"+{t('currency_symbol')}{real_time_bill['estimated_monthly_cost'] - plan_limit:.2f}"
                    delta_color = "inverse"
                else:
                    delta_text = f"-{t('currency_symbol')}{plan_limit - real_time_bill['estimated_monthly_cost']:.2f}"
                    delta_color = "normal"
                
                st.metric(
                    t('plan_difference'),
                    f"{t('currency_symbol')}{abs(real_time_bill['estimated_monthly_cost'] - plan_limit):.2f}",
                    delta=delta_text,
                    delta_color=delta_color
                )
            
            # 使用量詳細分解
            st.markdown(f"### 📊 {t('usage_breakdown')}")
            
            usage_summary = real_time_bill['usage_summary']
            cost_breakdown = real_time_bill['cost_breakdown']
            
            if usage_summary:
                # 創建使用量表格
                usage_df = pd.DataFrame([
                    {
                        t('service_type'): service_type,
                        t('usage'): quantity,
                        t('unit_cost'): f"{t('currency_symbol')}{cost_breakdown.get(service_type, 0) / max(quantity, 1):.4f}",
                        t('total_cost'): f"{t('currency_symbol')}{cost_breakdown.get(service_type, 0):.2f}"
                    }
                    for service_type, quantity in usage_summary.items()
                ])
                
                st.dataframe(usage_df, use_container_width=True)
                
                # 成本分佈餅圖
                if cost_breakdown:
                    fig = px.pie(
                        values=list(cost_breakdown.values()),
                        names=list(cost_breakdown.keys()),
                        title=t('cost_distribution')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(t('no_data'))
        
        else:
            st.info(t('no_data'))
        
        # 定價信息
        st.markdown(f"### 💎 {t('subscription_plans')}")
        
        pricing_info = self.billing_engine.get_pricing_info()
        plans = pricing_info['subscription_plans']
        
        plan_cols = st.columns(len(plans))
        
        for i, (plan_key, plan_data) in enumerate(plans.items()):
            with plan_cols[i]:
                current_plan = st.session_state.customer_info.get('plan_type', '') == plan_key
                plan_name = get_plan_translation(plan_key)
                
                if current_plan:
                    st.success(f"🌟 **{plan_name}** ({t('current_plan')})")
                else:
                    st.info(f"**{plan_name}**")
                
                st.write(f"**{t('monthly_fee')}**: {t('currency_symbol')}{plan_data['monthly_price']:.2f}")
                st.write(f"**{t('included')}**: {plan_data['included_predictions']:,}{t('predictions_unit')}")
                st.write(f"**{t('rate_limits')}**: {plan_data['max_requests_per_minute']}/{t('per_minute')}")
                
                if not current_plan:
                    if st.button(f"{t('upgrade_to')} {plan_name}", key=f"upgrade_{plan_key}"):
                        st.info(t('upgrade_feature_developing'))
    
    def _show_sla_status(self):
        """顯示SLA狀態"""
        st.markdown(f"## 🎯 {t('sla_title')}")
        
        # 獲取SLA狀態
        sla_status = self.sla_manager.get_current_sla_status()
        
        # 整體狀態
        overall_status = sla_status.get('overall_status', 'unknown')
        
        if overall_status == 'healthy':
            st.success(f"🟢 {t('all_services_normal')}")
        else:
            st.error(f"🔴 {t('service_issues')}")
        
        # SLA指標
        st.markdown(f"### 📊 {t('sla_metrics')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uptime_24h = sla_status.get('uptime_24h', 0)
            color = "normal" if uptime_24h >= 99.9 else "inverse"
            st.metric(
                t('uptime_24h'),
                f"{uptime_24h:.2f}{t('percentage')}",
                delta=f"{uptime_24h - 99.9:.2f}{t('percentage')}",
                delta_color=color
            )
        
        with col2:
            uptime_30d = sla_status.get('uptime_30d', 0)
            color = "normal" if uptime_30d >= 99.9 else "inverse"
            st.metric(
                t('uptime_30d'),
                f"{uptime_30d:.2f}{t('percentage')}",
                delta=f"{uptime_30d - 99.9:.2f}{t('percentage')}",
                delta_color=color
            )
        
        with col3:
            avg_response_time = sla_status.get('avg_response_time_1h', 0)
            if avg_response_time:
                color = "normal" if avg_response_time <= 500 else "inverse"
                st.metric(
                    t('avg_response_time'),
                    f"{avg_response_time:.0f}{t('milliseconds')}",
                    delta=f"{avg_response_time - 500:.0f}{t('milliseconds')}",
                    delta_color=color
                )
            else:
                st.metric(t('avg_response_time'), "N/A")
        
        # SLA目標
        st.markdown(f"### 🎯 {t('sla_commitments')}")
        
        sla_targets = sla_status.get('sla_targets', {})
        
        if sla_targets:
            for metric_name, target_data in sla_targets.items():
                with st.expander(f"📋 {target_data.get('description', metric_name)}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**{t('target_value')}**: {target_data.get('target_value', 'N/A')} {target_data.get('unit', '')}")
                        st.write(f"**{t('warning_threshold')}**: {target_data.get('threshold_warning', 'N/A')} {target_data.get('unit', '')}")
                    
                    with col2:
                        st.write(f"**{t('breach_threshold')}**: {target_data.get('threshold_breach', 'N/A')} {target_data.get('unit', '')}")
                        measurement_period = target_data.get('measurement_period', 0)
                        if measurement_period:
                            hours = measurement_period / 3600
                            if hours >= 24:
                                period_str = f"{hours/24:.1f}{t('days')}"
                            else:
                                period_str = f"{hours:.1f}{t('hours')}"
                            st.write(f"**{t('measurement_period')}**: {period_str}")
        
        # 當前健康檢查
        current_health = sla_status.get('current_health')
        if current_health:
            st.markdown(f"### 🏥 {t('current_health_check')}")
            
            endpoints = current_health.get('endpoints', {})
            if endpoints:
                for endpoint, status in endpoints.items():
                    healthy = status.get('healthy', False)
                    response_time = status.get('response_time_ms', 0)
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{endpoint}**")
                    
                    with col2:
                        if healthy:
                            st.success(f"🟢 {t('normal')}")
                        else:
                            st.error(f"🔴 {t('abnormal')}")
                    
                    with col3:
                        if response_time > 0:
                            st.write(f"{response_time:.0f}{t('milliseconds')}")
                        else:
                            st.write("N/A")
    
    def _show_account_settings(self):
        """顯示帳戶設定"""
        st.markdown(f"## ⚙️ {t('account_settings_title')}")
        
        customer_info = st.session_state.customer_info
        
        # 基本信息
        st.markdown(f"### 👤 {t('basic_information')}")
        
        with st.form("account_info"):
            col1, col2 = st.columns(2)
            
            with col1:
                email = st.text_input(t('email'), value=customer_info.get('email', ''))
                company_name = st.text_input(t('company_name'), value=customer_info.get('company_name', ''))
            
            with col2:
                plan_options = ['free', 'basic', 'professional', 'enterprise']
                current_plan = customer_info.get('plan_type', 'free')
                plan_type = st.selectbox(
                    t('subscription_plan'),
                    options=plan_options,
                    format_func=lambda x: get_plan_translation(x),
                    index=plan_options.index(current_plan) if current_plan in plan_options else 0
                )
            
            if st.form_submit_button(t('update_info')):
                st.success(t('account_updated'))
        
        # 安全設定
        st.markdown(f"### 🔒 {t('security_settings')}")
        
        with st.form("security_settings"):
            st.write(f"**{t('change_password')}**")
            current_password = st.text_input(t('current_password'), type="password")
            new_password = st.text_input(t('new_password'), type="password")
            confirm_password = st.text_input(t('confirm_password'), type="password")
            
            if st.form_submit_button(t('update_password')):
                if new_password == confirm_password:
                    st.success(t('password_updated'))
                else:
                    st.error(t('password_mismatch'))
        
        # 通知設定
        st.markdown(f"### 🔔 {t('notification_settings')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox(t('usage_alerts'), value=True)
            st.checkbox(t('billing_notifications'), value=True)
            st.checkbox(t('security_alerts'), value=True)
        
        with col2:
            st.checkbox(t('maintenance_notifications'), value=True)
            st.checkbox(t('feature_announcements'), value=False)
            st.checkbox(t('marketing_info'), value=False)
        
        # 危險區域
        st.markdown(f"### ⚠️ {t('danger_zone')}")
        
        with st.expander(f"🗑️ {t('delete_account')}", expanded=False):
            st.error(t('delete_account_warning'))
            
            delete_confirmation = st.text_input(
                t('delete_confirmation'),
                placeholder="DELETE"
            )
            
            if st.button(t('permanently_delete'), type="primary"):
                if delete_confirmation == "DELETE":
                    st.error(t('delete_feature_developing'))
                else:
                    st.error(t('confirm_delete_text'))
    
    def _authenticate_customer(self, email: str, password: str) -> bool:
        """客戶認證"""
        if email == "demo@example.com" and password == "demo123":
            st.session_state.logged_in = True
            st.session_state.customer_id = "demo_customer"
            st.session_state.customer_info = {
                'email': email,
                'company_name': 'Demo Company',
                'plan_type': 'professional'
            }
            return True
        
        return False
    
    def _register_customer(self, email: str, password: str) -> Optional[str]:
        """註冊新客戶"""
        try:
            customer_id = self.api_key_manager.database.create_customer(
                email=email,
                company_name="",
                plan_type="free"
            )
            return customer_id
        except Exception as e:
            print(f"Registration error: {e}")
            return None

def main():
    """主函數"""
    portal = InternationalCustomerPortal()
    portal.run()

if __name__ == "__main__":
    main()