"""
灵敏度分析 & 影子价格 交互式演示
医疗资源分配案例：普通病床 vs ICU 病床
约束：医护人员、呼吸机、防护服
目标：最大化收治收益
"""
import streamlit as st
import pulp
import plotly.graph_objects as go
import numpy as np

# ================= 页面配置 =================
st.set_page_config(page_title="灵敏度分析仪表盘", layout="wide")
st.title("🏥 医疗资源分配的灵敏度分析 & 影子价格")
st.markdown("拖动下方滑块改变**呼吸机数量**，观察最优解和影子价格如何实时变化。")

# ================= 1. 定义可重用的求解函数 =================
def solve_lp(b_ventilator):
    """
    根据呼吸机数量 b_ventilator 求解线性规划
    返回：目标值, x1(普通病床), x2(ICU病床), 约束字典(用于提取影子价格), 变量字典(用于缩减成本)
    """
    prob = pulp.LpProblem("Medical", pulp.LpMaximize)

    # 决策变量
    x1 = pulp.LpVariable("普通病床", lowBound=0, cat="Continuous")
    x2 = pulp.LpVariable("ICU病床", lowBound=0, cat="Continuous")

    # 目标函数：每人收治收益（元）
    prob += 3000 * x1 + 8000 * x2, "总收益"

    # 约束条件（注意给每个约束命名，方便后面提取影子价格）
    prob += 2 * x1 + 5 * x2 <= 200, "医护人员工时"
    prob += 1 * x1 + 2 * x2 <= b_ventilator, "呼吸机数量"  # 要调的资源
    prob += 3 * x1 + 6 * x2 <= 350, "防护服库存"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 提取结果
    obj_val = pulp.value(prob.objective)
    x1_val = pulp.value(x1)
    x2_val = pulp.value(x2)

    # 约束字典和变量字典方便后面直接拿影子价格与缩减成本
    constraints = {name: c for name, c in prob.constraints.items()}
    variables = {v.name: v for v in prob.variables()}

    return obj_val, x1_val, x2_val, constraints, variables

# ================= 2. 侧边栏滑块 =================
st.sidebar.header("🔧 可调参数")
ventilator_slider = st.sidebar.slider(
    "呼吸机数量 (B)", min_value=10, max_value=200, value=50, step=5,
    help="增加呼吸机相当于放松约束右端项，观察边际收益"
)

# 实时求解当前滑块值
obj, x1, x2, cons, vars_ = solve_lp(ventilator_slider)

# ================= 3. 核心指标卡片显示 =================
st.markdown("---")
st.subheader("📊 当前最优解")
col1, col2, col3 = st.columns(3)
col1.metric("最大总收益（元）", f"{obj:,.0f}")
col2.metric("普通病床收治量", f"{x1:.1f}")
col3.metric("ICU病床收治量", f"{x2:.1f}")

# ================= 4. 影子价格 & 缩减成本 =================
st.markdown("---")
st.subheader("💰 影子价格（对偶值）与缩减成本")

# 提取呼吸机约束的影子价格
shadow_ventilator = cons["呼吸机数量"].pi  # PuLP的.pi就是影子价格
st.info(f"**呼吸机的影子价格** = {shadow_ventilator:.2f} 元/台 → 每多一台呼吸机，总收益能增加这么多")

# 抽出所有约束的影子价格做个表
shadow_data = {}
for name, c in cons.items():
    shadow_data[name] = [c.pi, c.slack]  # 影子价格 和 松弛量
st.dataframe(
    {"约束": shadow_data.keys(),
     "影子价格": [v[0] for v in shadow_data.values()],
     "松弛量（剩余资源）": [v[1] for v in shadow_data.values()]},
    use_container_width=True
)

# 缩减成本：目标系数的灵敏度（非基变量进入基的代价）
st.subheader("🔻 缩减成本（Reduced Cost）")
rc_data = {}
for name, v in vars_.items():
    rc_data[name] = [v.varValue, v.dj]  # 取值 和 缩减成本
st.dataframe(
    {"变量": rc_data.keys(),
     "当前取值": [v[0] for v in rc_data.values()],
     "缩减成本": [v[1] for v in rc_data.values()]},
    use_container_width=True
)
st.caption("缩减成本为0的变量当前在最优基中；非零值表示该变量要想被采用，其收益系数至少需要增加的量。")

# ================= 5. 灵敏度曲线（影子价格的有效区间）=================
st.markdown("---")
st.subheader("📈 影子价格的有效区间 —— 利润随资源变化的曲线")

# 在全范围内计算一系列点
b_range = np.arange(10, 201, 5)
objs = []
for b in b_range:
    o, _, _, _, _ = solve_lp(b)
    objs.append(o)

# 用 plotly 画图
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=b_range, y=objs, mode='lines+markers',
    name='最优利润',
    hovertemplate='呼吸机:%{x}台<br>利润:%{y:,.0f}元'
))

# 标出当前滑块位置
fig.add_vline(x=ventilator_slider, line_dash="dash", line_color="red",
              annotation_text=f"当前B={ventilator_slider}", annotation_position="top")
fig.add_hline(y=obj, line_dash="dot", line_color="grey",
              annotation_text=f"当前利润{obj:,.0f}", annotation_position="top right")

# 自动探测拐点（斜率变化超过阈值的点）—— 基发生变化的地方
slopes = np.diff(objs) / np.diff(b_range)  # 利润差分/资源差分
tolerance = 1.0  # 斜率变化容忍度
break_points = []
for i in range(1, len(slopes)):
    if abs(slopes[i] - slopes[i-1]) > tolerance:
        bp = b_range[i+1]  # 拐点对应资源量
        break_points.append(bp)

for bp in break_points:
    fig.add_vline(x=bp, line_dash="dot", line_color="orange",
                  annotation_text=f"拐点{bp}", annotation_position="bottom")

fig.update_layout(title="呼吸机数量 vs 最大利润",
                  xaxis_title="呼吸机数量 (B)",
                  yaxis_title="最大利润 (元)",
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

if break_points:
    st.success(f"影子价格（{shadow_ventilator:.2f}）在呼吸机数量 ∈ [{min(break_points)}, {max(break_points)}] 时保持不变。超出范围后最优基改变，影子价格会跳变。")
else:
    st.success("当前变化范围内，最优基结构未发生变化，影子价格恒定。")

# ================= 6. 龙卷风图：目标系数灵敏度 =================
st.markdown("---")
st.subheader("🌪️ 目标系数灵敏度：龙卷风图（Tornado）")
st.markdown("当目标系数上下浮动 ±20% 时，观察总利润的变化幅度。")

base_profit = 3000
base_icu = 8000
# 浮动比例
delta = 0.2

# 计算每个系数单独变化后的最优利润
results = {}
# 普通病床收益变化
for label, base_val, var_name in [("普通病床收益", 3000, "普通病床"), ("ICU床位收益", 8000, "ICU病床")]:
    profits = []
    for factor in [1-delta, 1+delta]:
        # 重建模型，修改目标系数
        prob = pulp.LpProblem("Tornado", pulp.LpMaximize)
        x1 = pulp.LpVariable("普通病床", lowBound=0)
        x2 = pulp.LpVariable("ICU病床", lowBound=0)
        # 按 factor 修改对应系数
        c1 = base_profit if var_name=="普通病床" else base_profit * (1 if factor==1 else 1) # 保持另一个不变
        # 上面写得不好，重写逻辑：
        if var_name == "普通病床":
            c1 = base_val * factor
            c2 = base_icu
        else:
            c1 = base_profit
            c2 = base_val * factor
        prob += c1 * x1 + c2 * x2
        prob += 2*x1 + 5*x2 <= 200
        prob += 1*x1 + 2*x2 <= ventilator_slider  # 固定当前资源
        prob += 3*x1 + 6*x2 <= 350
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        profits.append(pulp.value(prob.objective))
    results[label] = profits

# 绘制龙卷风图（水平条形图）
fig_tornado = go.Figure()
y_labels = list(results.keys())
for i, (label, (low, high)) in enumerate(results.items()):
    base_val_center = (low + high)/2
    fig_tornado.add_trace(go.Bar(
        y=[label],
        x=[high - base_val_center],
        orientation='h',
        marker_color='blue',
        name='+20% 影响'
    ))
    fig_tornado.add_trace(go.Bar(
        y=[label],
        x=[low - base_val_center],
        orientation='h',
        marker_color='red',
        name='-20% 影响'
    ))

fig_tornado.update_layout(
    barmode='relative',
    title="目标系数变化±20%对最优利润的影响（相对中心值）",
    xaxis_title="利润变化量",
    yaxis_title="参数",
    showlegend=False
)
st.plotly_chart(fig_tornado, use_container_width=True)

st.caption("条形越长，说明该目标系数的波动对总利润影响越大，决策时应重点关注。")

# ================= 7. 脚注 =================
st.markdown("---")
st.success("✨ 你已经亲手完成了一次完整的灵敏度分析实验！拖动滑块可重新运行整个分析。")