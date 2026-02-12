from py2neo import Graph
from py2neo.errors import Neo4jError

# 1. 连接数据库
try:
    graph = Graph("neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"))
    print("✅ 成功连接到 Neo4j 数据库")
except Neo4jError as e:
    print(f"❌ 连接失败：{e}")
    exit()

# 2. 购车领域 Cypher 语句
cypher = """
// 清理旧数据（可选，仅用于演示重置）
// MATCH (n) DETACH DELETE n;

// 创建车辆节点 (包含品牌、型号、配置、价格、库存)
CREATE 
  (c1:Car {
    carId: 101,
    brand: "特斯拉",
    model: "Model Y",
    trim: "长续航全轮驱动版",
    year: 2024,
    price: 299900,
    engineType: "纯电动",
    rangeKM: 688,
    stock: 50,
    colorOptions: ["白色", "黑色", "深海蓝", "红色"]
  }),
  (c2:Car {
    carId: 102,
    brand: "比亚迪",
    model: "汉EV",
    trim: "冠军版 715KM尊享型",
    year: 2024,
    price: 219800,
    engineType: "纯电动",
    rangeKM: 715,
    stock: 80,
    colorOptions: ["银色", "黑色", "金鳞橙"]
  }),
  (c3:Car {
    carId: 103,
    brand: "丰田",
    model: "凯美瑞",
    trim: "2.5L 智能电混双擎 豪华版",
    year: 2024,
    price: 199800,
    engineType: "油电混动",
    rangeKM: 1000,
    stock: 30,
    colorOptions: ["白色", "灰色", "蓝色"]
  }),

  // 创建车主节点
  (u1:User {
    userId: 1001,
    name: "王强",
    age: 35,
    city: "深圳",
    driverLicenseType: "C1",
    experienceYears: 8
  }),
  (u2:User {
    userId: 1002,
    name: "赵敏",
    age: 28,
    city: "杭州",
    driverLicenseType: "C1",
    experienceYears: 3
  }),

  // 创建经销商节点
  (d1:Dealership {
    dealerId: 5001,
    name: "深圳南山特斯拉中心",
    location: "广东省深圳市南山区",
    rating: 4.9
  }),
  (d2:Dealership {
    dealerId: 5002,
    name: "杭州滨江比亚迪4S店",
    location: "浙江省杭州市滨江区",
    rating: 4.7
  }),

  // 创建金融方案节点 (展示购车特有的贷款/全款选项)
  (f1:FinancePlan {
    planId: 9001,
    type: "零首付分期",
    durationMonths: 36,
    annualRate: 2.99,
    description: "前12个月仅还利息"
  }),
  (f2:FinancePlan {
    planId: 9002,
    type: "全款购车",
    durationMonths: 0,
    annualRate: 0,
    description: "一次性付清，享2000元保险补贴"
  }),

  // 创建关系
  // 王强购买了特斯拉 Model Y，并使用了零首付分期方案，由深圳南山特斯拉中心销售
  (u1)-[:PURCHASED {
    purchaseDate: "2024-11-15", 
        totalPrice: 299900, 
        status: "已交付"
  }]->(c1),
  (u1)-[:USED_PLAN]->(f1),
  (c1)-[:SOLD_BY]->(d1),

  // 赵敏购买了比亚迪汉EV，选择了全款方案，由杭州滨江比亚迪4S店销售
  (u2)-[:PURCHASED {
    purchaseDate: "2024-11-20", 
        totalPrice: 219800, 
        status: "已交付"
  }]->(c2),
  (u2)-[:USED_PLAN]->(f2),
  (c2)-[:SOLD_BY]->(d2),

  // 额外关系：用户对车辆的意向（比如试驾过）
  (u1)-[:INTERESTED_IN {testDriveDate: "2024-11-10"}]->(c3);

"""

# 3. 执行语句
try:
    graph.run(cypher)
    print("📝 购车数据创建及关系绑定成功！")
except Neo4jError as e:
    print(f"❌ 执行失败：{e}")

# 4. 验证查询：展示王强买了什么车，用了什么方案
verify_cypher = """
MATCH (u:User {name: '王强'})-[:PURCHASED]->(car:Car)<-[:SOLD_BY]-(dealer:Dealership)
RETURN u.name AS 用户, car.model AS 车型, car.trim AS 配置, dealer.name AS 经销商
"""
results = graph.run(verify_cypher).data()
print("\n--- 📋 购车记录验证 ---")
for record in results:
    print(f"👨‍💼 {record['用户']} 购买了 {record['车型']} ({record['配置']})")
    print(f"    🏬 经销商: {record['经销商']}")