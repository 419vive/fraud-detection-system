"""
客戶門戶和管理面板 - IEEE-CIS 詐騙檢測服務
提供客戶自助服務門戶和管理面板功能
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

class CustomerPortal:
    """客戶門戶類"""
    
    def __init__(self):
        self.billing_engine = create_billing_engine()
        self.api_key_manager = create_api_key_manager()
        self.sla_manager = create_sla_manager()
        self.usage_monitoring = create_usage_monitoring_system(
            self.billing_engine, self.api_key_manager
        )
        
        # 初始化session state
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'customer_id' not in st.session_state:
            st.session_state.customer_id = None
        if 'customer_info' not in st.session_state:
            st.session_state.customer_info = None
    
    def run(self):
        """運行客戶門戶應用"""
        st.set_page_config(
            page_title="詐騙檢測API - 客戶門戶",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
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
        </style>
        """, unsafe_allow_html=True)
        
        # 主標題
        st.markdown("""
        <div class="main-header">
            <h1>🛡️ 詐騙檢測API - 客戶門戶</h1>
            <p>管理您的API密鑰、監控使用量、查看計費信息</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 檢查登入狀態
        if not st.session_state.logged_in:
            self._show_login_page()
        else:
            self._show_main_portal()
    
    def _show_login_page(self):
        """顯示登入頁面"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔐 客戶登入")
            
            with st.form("login_form"):
                email = st.text_input("📧 電子郵件")
                password = st.text_input("🔒 密碼", type="password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_button = st.form_submit_button("登入", use_container_width=True)
                with col_b:
                    register_button = st.form_submit_button("註冊", use_container_width=True)
                
                if login_button:
                    if self._authenticate_customer(email, password):
                        st.success("登入成功！")
                        st.rerun()
                    else:
                        st.error("登入失敗，請檢查您的認證信息")
                
                if register_button:
                    if email and password:
                        customer_id = self._register_customer(email, password)
                        if customer_id:
                            st.success("註冊成功！請登入")
                        else:
                            st.error("註冊失敗，該電子郵件可能已被使用")
                    else:
                        st.error("請填寫所有欄位")
            
            # 演示帳號
            st.markdown("---")
            st.markdown("### 🎯 演示帳號")
            if st.button("使用演示帳號登入", use_container_width=True):
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
            st.markdown("### 👤 客戶信息")
            customer_info = st.session_state.customer_info
            st.write(f"**公司**: {customer_info.get('company_name', 'N/A')}")
            st.write(f"**方案**: {customer_info.get('plan_type', 'N/A').title()}")
            st.write(f"**客戶ID**: {st.session_state.customer_id}")
            
            st.markdown("---")
            
            # 導航菜單
            page = st.selectbox(
                "📋 選擇頁面",
                [
                    "儀表板",
                    "API密鑰管理", 
                    "使用量監控",
                    "計費信息",
                    "SLA狀態",
                    "帳戶設定"
                ]
            )
            
            st.markdown("---")
            
            if st.button("登出", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.customer_id = None
                st.session_state.customer_info = None
                st.rerun()
        
        # 主要內容區域
        if page == "儀表板":
            self._show_dashboard()
        elif page == "API密鑰管理":
            self._show_api_key_management()
        elif page == "使用量監控":
            self._show_usage_monitoring()
        elif page == "計費信息":
            self._show_billing_info()
        elif page == "SLA狀態":
            self._show_sla_status()
        elif page == "帳戶設定":
            self._show_account_settings()
    
    def _show_dashboard(self):
        """顯示儀表板"""
        st.markdown("## 📊 客戶儀表板")
        
        customer_id = st.session_state.customer_id
        
        # 獲取儀表板數據
        dashboard_data = self.usage_monitoring.get_customer_usage_dashboard(customer_id)
        real_time_bill = dashboard_data.get('real_time_bill')
        metrics = dashboard_data.get('metrics', {})
        alerts = dashboard_data.get('alerts', [])
        
        # 頂部指標卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>💰 本月費用</h4>
                <h2>${:.2f}</h2>
                <p>預估月度成本</p>
            </div>
            """.format(
                real_time_bill.get('estimated_monthly_cost', 0) if real_time_bill else 0
            ), unsafe_allow_html=True)
        
        with col2:
            total_requests = sum(real_time_bill.get('usage_summary', {}).values()) if real_time_bill else 0
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 本月請求</h4>
                <h2>{total_requests:,}</h2>
                <p>API調用次數</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_response_time = metrics.get('response_time', {}).get('average', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚡ 平均響應時間</h4>
                <h2>{avg_response_time:.0f}ms</h2>
                <p>最近1小時</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            api_keys_count = dashboard_data.get('api_keys_count', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔑 API密鑰</h4>
                <h2>{api_keys_count}</h2>
                <p>已創建密鑰數量</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 圖表區域
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 使用量趨勢")
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
                    title="本月API使用量分布",
                    xaxis_title="服務類型",
                    yaxis_title="調用次數",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("暫無使用量數據")
        
        with col2:
            st.markdown("### ⚡ 性能指標")
            if metrics.get('response_time'):
                response_metrics = metrics['response_time']
                
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=response_metrics.get('average', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "平均響應時間 (ms)"},
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
                st.info("暫無性能數據")
        
        # 警報區域
        if alerts:
            st.markdown("### 🚨 活躍警報")
            for alert in alerts:
                alert_type = alert.get('alert_type', '未知')
                message = alert.get('message', '無詳細信息')
                
                if 'high' in alert_type.lower():
                    st.error(f"**{alert_type}**: {message}")
                elif 'medium' in alert_type.lower():
                    st.warning(f"**{alert_type}**: {message}")
                else:
                    st.info(f"**{alert_type}**: {message}")
        else:
            st.success("🟢 目前沒有活躍警報")
    
    def _show_api_key_management(self):
        """顯示API密鑰管理"""
        st.markdown("## 🔑 API密鑰管理")
        
        customer_id = st.session_state.customer_id
        
        # 獲取現有密鑰
        api_keys = self.api_key_manager.get_customer_keys(customer_id)
        
        # 創建新密鑰區域
        with st.expander("➕ 創建新API密鑰", expanded=False):
            with st.form("create_api_key"):
                col1, col2 = st.columns(2)
                
                with col1:
                    key_name = st.text_input("密鑰名稱", placeholder="例如：生產API密鑰")
                    permissions = st.multiselect(
                        "權限",
                        options=[p.value for p in Permission],
                        default=['predict', 'read']
                    )
                
                with col2:
                    expires_in_days = st.number_input("有效期（天）", min_value=1, max_value=3650, value=365)
                    allowed_ips = st.text_area(
                        "允許的IP地址（每行一個）",
                        placeholder="192.168.1.1\n10.0.0.1",
                        height=100
                    )
                
                if st.form_submit_button("創建API密鑰", use_container_width=True):
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
                            
                            st.success("API密鑰創建成功！")
                            st.code(f"API密鑰: {raw_key}")
                            st.warning("⚠️ 請妥善保管您的API密鑰，我們不會再次顯示完整密鑰")
                            
                            # 刷新頁面
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"創建API密鑰失敗: {str(e)}")
                    else:
                        st.error("請填寫密鑰名稱")
        
        # 現有密鑰列表
        st.markdown("### 📋 現有API密鑰")
        
        if api_keys:
            for api_key in api_keys:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        status_color = "🟢" if api_key.is_valid() else "🔴"
                        st.write(f"{status_color} **{api_key.name}**")
                        st.caption(f"ID: {api_key.id[:8]}...")
                    
                    with col2:
                        st.write(f"**權限**: {len(api_key.permissions)}")
                        permissions_str = ", ".join([p.value for p in api_key.permissions])
                        st.caption(permissions_str)
                    
                    with col3:
                        st.write(f"**使用次數**: {api_key.usage_count:,}")
                        if api_key.last_used_at:
                            st.caption(f"最後使用: {api_key.last_used_at.strftime('%Y-%m-%d')}")
                        else:
                            st.caption("從未使用")
                    
                    with col4:
                        if st.button("🗑️", key=f"delete_{api_key.id}", help="撤銷密鑰"):
                            if self.api_key_manager.revoke_api_key(api_key.id):
                                st.success("API密鑰已撤銷")
                                st.rerun()
                            else:
                                st.error("撤銷失敗")
                    
                    # 詳細信息
                    with st.expander(f"📊 {api_key.name} - 詳細信息"):
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            st.write("**基本信息**")
                            st.write(f"創建日期: {api_key.created_at.strftime('%Y-%m-%d %H:%M')}")
                            if api_key.expires_at:
                                st.write(f"到期日期: {api_key.expires_at.strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"狀態: {api_key.status.value}")
                        
                        with info_col2:
                            st.write("**速率限制**")
                            st.write(f"每分鐘: {api_key.rate_limits.requests_per_minute}")
                            st.write(f"每小時: {api_key.rate_limits.requests_per_hour}")
                            st.write(f"每天: {api_key.rate_limits.requests_per_day}")
                        
                        if api_key.allowed_ips:
                            st.write("**允許的IP地址**")
                            for ip in api_key.allowed_ips:
                                st.code(ip)
                    
                    st.markdown("---")
        else:
            st.info("您還沒有創建任何API密鑰")
    
    def _show_usage_monitoring(self):
        """顯示使用量監控"""
        st.markdown("## 📈 使用量監控")
        
        customer_id = st.session_state.customer_id
        
        # 時間範圍選擇
        col1, col2 = st.columns(2)
        with col1:
            time_range = st.selectbox(
                "時間範圍",
                ["最近24小時", "最近7天", "最近30天", "自定義"]
            )
        
        with col2:
            if time_range == "自定義":
                date_range = st.date_input(
                    "選擇日期範圍",
                    value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
                    max_value=datetime.now().date()
                )
        
        # 獲取使用統計
        if time_range == "最近24小時":
            days = 1
        elif time_range == "最近7天":
            days = 7
        elif time_range == "最近30天":
            days = 30
        else:
            days = 7  # 默認
        
        usage_stats = self.api_key_manager.get_usage_statistics(customer_id, days)
        
        # 顯示統計數據
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "總請求數",
                f"{usage_stats.get('total_requests', 0):,}",
                delta=None
            )
        
        with col2:
            success_rate = usage_stats.get('success_rate', 0)
            st.metric(
                "成功率",
                f"{success_rate:.1f}%",
                delta=f"{success_rate - 95:.1f}%" if success_rate > 0 else None
            )
        
        with col3:
            avg_response_time = usage_stats.get('average_response_time_ms', 0)
            st.metric(
                "平均響應時間",
                f"{avg_response_time:.0f}ms",
                delta=None
            )
        
        # 狀態碼分佈圖
        if usage_stats.get('status_distribution'):
            st.markdown("### 📊 HTTP狀態碼分佈")
            
            status_data = usage_stats['status_distribution']
            
            fig = px.pie(
                values=list(status_data.values()),
                names=[f"HTTP {code}" for code in status_data.keys()],
                title="請求狀態分佈"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # 實時指標
        st.markdown("### ⚡ 實時指標")
        
        dashboard_data = self.usage_monitoring.get_customer_usage_dashboard(customer_id)
        metrics = dashboard_data.get('metrics', {})
        
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'response_time' in metrics:
                    rt_metrics = metrics['response_time']
                    st.write("**響應時間統計 (最近1小時)**")
                    st.write(f"平均: {rt_metrics.get('average', 0):.0f}ms")
                    st.write(f"最小: {rt_metrics.get('min', 0):.0f}ms")
                    st.write(f"最大: {rt_metrics.get('max', 0):.0f}ms")
                    st.write(f"P95: {rt_metrics.get('p95', 0):.0f}ms")
            
            with col2:
                if 'throughput' in metrics:
                    tp_metrics = metrics['throughput']
                    st.write("**吞吐量統計**")
                    st.write(f"每分鐘請求數: {tp_metrics.get('requests_per_minute', 0):.1f}")
                    st.write(f"總請求數: {tp_metrics.get('count', 0)}")
        else:
            st.info("暫無實時指標數據")
    
    def _show_billing_info(self):
        """顯示計費信息"""
        st.markdown("## 💰 計費信息")
        
        customer_id = st.session_state.customer_id
        
        # 獲取實時帳單
        dashboard_data = self.usage_monitoring.get_customer_usage_dashboard(customer_id)
        real_time_bill = dashboard_data.get('real_time_bill')
        
        if real_time_bill:
            # 當前計費期間
            st.markdown("### 📅 當前計費期間")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**開始日期**: {real_time_bill['current_period_start'][:10]}")
                st.write(f"**結束日期**: {real_time_bill['current_period_end'][:10]}")
            
            with col2:
                st.write(f"**上次更新**: {real_time_bill['last_updated'][:19]}")
            
            # 費用摘要
            st.markdown("### 💸 費用摘要")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "當前費用",
                    f"${real_time_bill['total_cost']:.2f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "預估月度費用",
                    f"${real_time_bill['estimated_monthly_cost']:.2f}",
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
                    delta_text = f"+${real_time_bill['estimated_monthly_cost'] - plan_limit:.2f}"
                    delta_color = "inverse"
                else:
                    delta_text = f"-${plan_limit - real_time_bill['estimated_monthly_cost']:.2f}"
                    delta_color = "normal"
                
                st.metric(
                    "與計劃差額",
                    f"${abs(real_time_bill['estimated_monthly_cost'] - plan_limit):.2f}",
                    delta=delta_text,
                    delta_color=delta_color
                )
            
            # 使用量詳細分解
            st.markdown("### 📊 使用量分解")
            
            usage_summary = real_time_bill['usage_summary']
            cost_breakdown = real_time_bill['cost_breakdown']
            
            if usage_summary:
                # 創建使用量表格
                usage_df = pd.DataFrame([
                    {
                        '服務類型': service_type,
                        '使用量': quantity,
                        '單位成本': f"${cost_breakdown.get(service_type, 0) / max(quantity, 1):.4f}",
                        '總成本': f"${cost_breakdown.get(service_type, 0):.2f}"
                    }
                    for service_type, quantity in usage_summary.items()
                ])
                
                st.dataframe(usage_df, use_container_width=True)
                
                # 成本分佈餅圖
                if cost_breakdown:
                    fig = px.pie(
                        values=list(cost_breakdown.values()),
                        names=list(cost_breakdown.keys()),
                        title="成本分佈"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("本期間內暫無使用量")
        
        else:
            st.info("暫無計費數據")
        
        # 定價信息
        st.markdown("### 💎 訂閱方案")
        
        pricing_info = self.billing_engine.get_pricing_info()
        plans = pricing_info['subscription_plans']
        
        plan_cols = st.columns(len(plans))
        
        for i, (plan_key, plan_data) in enumerate(plans.items()):
            with plan_cols[i]:
                current_plan = st.session_state.customer_info.get('plan_type', '') == plan_key
                
                if current_plan:
                    st.success(f"🌟 **{plan_data['name']}** (當前方案)")
                else:
                    st.info(f"**{plan_data['name']}**")
                
                st.write(f"**月費**: ${plan_data['monthly_price']:.2f}")
                st.write(f"**包含**: {plan_data['included_predictions']:,} 次預測")
                st.write(f"**速率限制**: {plan_data['max_requests_per_minute']}/分鐘")
                
                if not current_plan:
                    if st.button(f"升級到 {plan_data['name']}", key=f"upgrade_{plan_key}"):
                        # 這裡可以實現升級邏輯
                        st.info("升級功能開發中...")
    
    def _show_sla_status(self):
        """顯示SLA狀態"""
        st.markdown("## 🎯 服務等級協議 (SLA)")
        
        # 獲取SLA狀態
        sla_status = self.sla_manager.get_current_sla_status()
        
        # 整體狀態
        overall_status = sla_status.get('overall_status', 'unknown')
        
        if overall_status == 'healthy':
            st.success("🟢 所有服務運行正常")
        else:
            st.error("🔴 服務存在問題")
        
        # SLA指標
        st.markdown("### 📊 SLA指標")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uptime_24h = sla_status.get('uptime_24h', 0)
            color = "normal" if uptime_24h >= 99.9 else "inverse"
            st.metric(
                "24小時正常運行時間",
                f"{uptime_24h:.2f}%",
                delta=f"{uptime_24h - 99.9:.2f}%",
                delta_color=color
            )
        
        with col2:
            uptime_30d = sla_status.get('uptime_30d', 0)
            color = "normal" if uptime_30d >= 99.9 else "inverse"
            st.metric(
                "30天正常運行時間",
                f"{uptime_30d:.2f}%",
                delta=f"{uptime_30d - 99.9:.2f}%",
                delta_color=color
            )
        
        with col3:
            avg_response_time = sla_status.get('avg_response_time_1h', 0)
            if avg_response_time:
                color = "normal" if avg_response_time <= 500 else "inverse"
                st.metric(
                    "平均響應時間",
                    f"{avg_response_time:.0f}ms",
                    delta=f"{avg_response_time - 500:.0f}ms",
                    delta_color=color
                )
            else:
                st.metric("平均響應時間", "N/A")
        
        # SLA目標
        st.markdown("### 🎯 SLA承諾")
        
        sla_targets = sla_status.get('sla_targets', {})
        
        if sla_targets:
            for metric_name, target_data in sla_targets.items():
                with st.expander(f"📋 {target_data.get('description', metric_name)}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**目標值**: {target_data.get('target_value', 'N/A')} {target_data.get('unit', '')}")
                        st.write(f"**警告閾值**: {target_data.get('threshold_warning', 'N/A')} {target_data.get('unit', '')}")
                    
                    with col2:
                        st.write(f"**違約閾值**: {target_data.get('threshold_breach', 'N/A')} {target_data.get('unit', '')}")
                        measurement_period = target_data.get('measurement_period', 0)
                        if measurement_period:
                            hours = measurement_period / 3600
                            if hours >= 24:
                                period_str = f"{hours/24:.1f} 天"
                            else:
                                period_str = f"{hours:.1f} 小時"
                            st.write(f"**測量期間**: {period_str}")
        
        # 當前健康檢查
        current_health = sla_status.get('current_health')
        if current_health:
            st.markdown("### 🏥 當前健康檢查")
            
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
                            st.success("🟢 正常")
                        else:
                            st.error("🔴 異常")
                    
                    with col3:
                        if response_time > 0:
                            st.write(f"{response_time:.0f}ms")
                        else:
                            st.write("N/A")
    
    def _show_account_settings(self):
        """顯示帳戶設定"""
        st.markdown("## ⚙️ 帳戶設定")
        
        customer_info = st.session_state.customer_info
        
        # 基本信息
        st.markdown("### 👤 基本信息")
        
        with st.form("account_info"):
            col1, col2 = st.columns(2)
            
            with col1:
                email = st.text_input("電子郵件", value=customer_info.get('email', ''))
                company_name = st.text_input("公司名稱", value=customer_info.get('company_name', ''))
            
            with col2:
                plan_type = st.selectbox(
                    "訂閱方案",
                    options=['free', 'basic', 'professional', 'enterprise'],
                    index=['free', 'basic', 'professional', 'enterprise'].index(customer_info.get('plan_type', 'free'))
                )
            
            if st.form_submit_button("更新信息"):
                # 這裡可以實現更新邏輯
                st.success("帳戶信息已更新")
        
        # 安全設定
        st.markdown("### 🔒 安全設定")
        
        with st.form("security_settings"):
            st.write("**變更密碼**")
            current_password = st.text_input("當前密碼", type="password")
            new_password = st.text_input("新密碼", type="password")
            confirm_password = st.text_input("確認新密碼", type="password")
            
            if st.form_submit_button("更新密碼"):
                if new_password == confirm_password:
                    st.success("密碼已更新")
                else:
                    st.error("新密碼與確認密碼不符")
        
        # 通知設定
        st.markdown("### 🔔 通知設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("使用量警報", value=True)
            st.checkbox("計費通知", value=True)
            st.checkbox("安全警報", value=True)
        
        with col2:
            st.checkbox("維護通知", value=True)
            st.checkbox("新功能公告", value=False)
            st.checkbox("營銷信息", value=False)
        
        # 危險區域
        st.markdown("### ⚠️ 危險區域")
        
        with st.expander("🗑️ 刪除帳戶", expanded=False):
            st.error("**警告**: 刪除帳戶將永久移除所有數據，此操作無法撤銷。")
            
            delete_confirmation = st.text_input(
                "輸入 'DELETE' 確認刪除帳戶",
                placeholder="DELETE"
            )
            
            if st.button("永久刪除帳戶", type="primary"):
                if delete_confirmation == "DELETE":
                    st.error("帳戶刪除功能開發中...")
                else:
                    st.error("請輸入 'DELETE' 確認刪除")
    
    def _authenticate_customer(self, email: str, password: str) -> bool:
        """客戶認證"""
        # 這裡應該實現真實的認證邏輯
        # 為了演示，我們使用簡化的認證
        
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
            # 創建客戶記錄
            customer_id = self.api_key_manager.database.create_customer(
                email=email,
                company_name="",
                plan_type="free"
            )
            
            return customer_id
        except Exception as e:
            print(f"註冊錯誤: {e}")
            return None

def main():
    """主函數"""
    portal = CustomerPortal()
    portal.run()

if __name__ == "__main__":
    main()