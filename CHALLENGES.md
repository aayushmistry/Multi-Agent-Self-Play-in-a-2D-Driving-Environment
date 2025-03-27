## 🚧 Challenges & Bottleneck Analysis

### ⚙️ Summary of Training Issues (based on `Final_10.csv` and reward plots):

| Metric              | Value (Avg) | Max | Notes |
|---------------------|-------------|-----|-------|
| 💡 Success Rate      | 0.089        | 0.20 | Only 1–2 agents succeed per episode |
| 💥 Collision Rate    | 0.27         | 1.00 | Collisions still frequent — needs coordination |
| 🏆 Total Reward      | -227         | 193 | High variance due to early failures |
| ↻ Episode Length    | ~4 steps     | —   | Quick terminations due to early goal or crashes |

---

### 🧠 Key Challenges Identified

#### 1. **Low Agent Success Rate**
- PPO learns for a **subset of agents only**
- Shared policy lacks per-agent role awareness

✅ Partial learning  
❗ No coordination

#### 2. **Frequent Collisions**
- Agents crowd near intersections and goal zones
- No yielding, path planning, or spatial awareness

✅ Some collision-free episodes  
❗ Still highly volatile

#### 3. **Reward Instability**
- Rewards range from -4500 to +190  
- High variance and early negative spikes

✅ Improved over time  
❗ Clipping, sparse reward delay early convergence

#### 4. **Short Episode Lifespan**
- Average of ~4 steps → agents crash or reach goal quickly
- Doesn't promote long-term strategy learning

---

### 🔧 Recommendations

| Fix Area           | Suggestions |
|--------------------|-------------|
| Success Rate       | Add multiple dynamic goal points per agent |
| Collision Avoidance| Penalize agent overlap, add “wait” action, improve lane logic |
| Stability          | Use reward normalization, adjust `clip_range`, or try curriculum learning |
| Coordination       | Include agent ID in observation space for shared policy differentiation |



