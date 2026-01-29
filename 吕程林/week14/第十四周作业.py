"""
汽车贷款Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义 ====================
# 银行存贷产品核心工具函数
def get_bank_products():
    """
    获取所有可用的银行存贷产品列表
    本质是给予一些产品的信息
    """
    products = [
        {
            "id": "loan_car_001",
            "name": "新车按揭贷款",
            "type": "贷款",
            "description": "用于购买新车的分期贷款，支持多种还款方式",
            "min_amount": 50000,
            "max_amount": 2000000,
            "min_years": 1,
            "max_years": 5,
            "base_rate": 3.85  # 年化利率（%）
        },
        {
            "id": "loan_car_002",
            "name": "二手车贷款",
            "type": "贷款",
            "description": "专为购买二手车设计的贷款产品，评估流程简便",
            "min_amount": 20000,
            "max_amount": 800000,
            "min_years": 1,
            "max_years": 3,
            "base_rate": 4.2  # 年化利率（%）
        },
        {
            "id": "loan_car_003",
            "name": "汽车抵押贷款",
            "type": "贷款",
            "description": "以自有车辆作为抵押物获取资金，审批快速",
            "min_amount": 30000,
            "max_amount": 1500000,
            "min_years": 1,
            "max_years": 4,
            "base_rate": 4.5  # 年化利率（%）
        },
        {
            "id": "loan_car_004",
            "name": "汽车消费贷款",
            "type": "贷款",
            "description": "用于汽车相关消费的信用贷款，无需抵押",
            "min_amount": 10000,
            "max_amount": 500000,
            "min_years": 1,
            "max_years": 3,
            "base_rate": 5.2  # 年化利率（%）
        }
    ]
    return json.dumps(products, ensure_ascii=False)

def get_product_detail(product_id: str):
    """
    获取指定产品的详细信息，其实本质就是介绍的更加详细

    Args:
        product_id: 产品ID
    """
    products = {
        "loan_car_001": {
            "id": "loan_car_001",
            "name": "新车按揭贷款",
            "type": "贷款",
            "description": "用于购买新车的分期贷款，支持多种还款方式，最长5年",
            "min_amount": 50000,
            "max_amount": 2000000,
            "term_options": [1, 2, 3, 4, 5],
            "rate_by_term": {1: 3.85, 2: 4.0, 3: 4.15, 4: 4.3, 5: 4.45},
            "repayment_method": "等额本息/等额本金",
            "age_limit": "18-60周岁",
            "credit_note": "利率会根据个人征信情况浮动±10%，首付款比例不低于20%"
        },
        "loan_car_002": {
            "id": "loan_car_002",
            "name": "二手车贷款",
            "type": "贷款",
            "description": "专为购买二手车设计的贷款产品，评估流程简便，最长3年",
            "min_amount": 20000,
            "max_amount": 800000,
            "term_options": [1, 2, 3],
            "rate_by_term": {1: 4.2, 2: 4.35, 3: 4.5},
            "repayment_method": "等额本息/等额本金",
            "age_limit": "18-60周岁",
            "credit_note": "利率会根据个人征信情况浮动±10%，首付款比例不低于30%"
        },
        "loan_car_003": {
            "id": "loan_car_003",
            "name": "汽车抵押贷款",
            "type": "贷款",
            "description": "以自有车辆作为抵押物获取资金，审批快速，最长4年",
            "min_amount": 30000,
            "max_amount": 1500000,
            "term_options": [1, 2, 3, 4],
            "rate_by_term": {1: 4.5, 2: 4.65, 3: 4.8, 4: 4.95},
            "repayment_method": "等额本息/等额本金",
            "age_limit": "18-60周岁",
            "credit_note": "利率会根据车辆评估价值和个人征信情况浮动±10%"
        },
        "loan_car_004": {
            "id": "loan_car_004",
            "name": "汽车消费贷款",
            "type": "贷款",
            "description": "用于汽车相关消费的信用贷款，无需抵押，最长3年",
            "min_amount": 10000,
            "max_amount": 500000,
            "term_options": [1, 2, 3],
            "rate_by_term": {1: 5.2, 2: 5.35, 3: 5.5},
            "repayment_method": "等额本息/等额本金",
            "age_limit": "18-60周岁",
            "credit_note": "利率会根据个人征信情况浮动±10%，可用于汽车维修、改装等消费"
        }
    }

    if product_id in products:
        return json.dumps(products[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)


def calculate_loan_repayment(product_id: str, loan_amount: int, years: float, repayment_method: str = "等额本息"):
    """
    计算车贷还款明细（本金、利息和总金额）

    Args:
        product_id: 产品ID
        loan_amount: 贷款金额（元）
        years: 贷款年限
        repayment_method: 还款方式（"等额本息"或"等额本金"）
    """
    # 获取产品详情
    product_detail = json.loads(get_product_detail(product_id))
    if "error" in product_detail:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)

    if product_detail["type"] != "贷款":
        return json.dumps({"error": "该产品不是贷款产品，无法计算还款"}, ensure_ascii=False)

    # 验证贷款金额是否在范围内
    if loan_amount < product_detail["min_amount"] or loan_amount > product_detail["max_amount"]:
        return json.dumps({
                              "error": f"贷款金额{loan_amount}元不在该产品范围内（{product_detail['min_amount']}-{product_detail['max_amount']}元）"},
                          ensure_ascii=False)

    # 验证年限是否在可选范围内
    if years not in product_detail["term_options"]:
        return json.dumps({"error": f"该产品不支持{years}年期限，可选期限：{product_detail['term_options']}"},
                          ensure_ascii=False)

    # 验证还款方式
    if repayment_method not in ["等额本息", "等额本金"]:
        return json.dumps({"error": "还款方式错误，可选值：'等额本息'或'等额本金'"}, ensure_ascii=False)

    # 获取对应年限的利率
    rate = product_detail["rate_by_term"][years] / 100
    monthly_rate = rate / 12
    total_months = int(years * 12)

    # 计算还款明细
    if repayment_method == "等额本息":
        # 等额本息计算公式：每月还款额 = [本金×月利率×(1+月利率)^总期数]÷[(1+月利率)^总期数-1]
        monthly_payment = loan_amount * monthly_rate * (1 + monthly_rate) ** total_months / (
                    (1 + monthly_rate) ** total_months - 1)
        total_repayment = round(monthly_payment * total_months, 2)
        total_interest = round(total_repayment - loan_amount, 2)

        # 生成每月还款明细
        monthly_details = []
        remaining_principal = loan_amount
        for month in range(1, total_months + 1):
            interest = round(remaining_principal * monthly_rate, 2)
            principal = round(monthly_payment - interest, 2)
            remaining_principal = round(remaining_principal - principal, 2)
            monthly_details.append({
                "month": month,
                "principal": principal,
                "interest": interest,
                "total_payment": round(principal + interest, 2),
                "remaining_principal": remaining_principal
            })
    else:
        # 等额本金计算：每月还固定本金，利息逐月递减
        monthly_principal = loan_amount / total_months
        total_interest = 0
        monthly_details = []
        remaining_principal = loan_amount

        for month in range(1, total_months + 1):
            interest = round(remaining_principal * monthly_rate, 2)
            total_interest += interest
            monthly_payment = round(monthly_principal + interest, 2)
            remaining_principal = round(remaining_principal - monthly_principal, 2)
            monthly_details.append({
                "month": month,
                "principal": round(monthly_principal, 2),
                "interest": interest,
                "total_payment": monthly_payment,
                "remaining_principal": remaining_principal
            })

        total_repayment = round(loan_amount + total_interest, 2)

    result = {
        "product_id": product_id,
        "product_name": product_detail["name"],
        "loan_amount": loan_amount,
        "years": years,
        "annual_rate": f"{rate * 100}%",
        "repayment_method": repayment_method,
        "total_principal": loan_amount,
        "total_interest": total_interest,
        "total_repayment": total_repayment,
        "monthly_payment_avg": round(total_repayment / total_months, 2),
        "monthly_details": monthly_details,
        "credit_note": product_detail.get("credit_note", "利率会根据个人征信情况浮动±10%")
    }

    return json.dumps(result, ensure_ascii=False)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_bank_products",
            "description": "获取所有可用的银行存贷产品列表，包括产品名称、类型、金额范围、年限范围、基础利率等基本信息",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "获取指定银行产品的详细信息，包括期限选项、不同期限利率、还款方式、年龄限制等",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "产品ID，例如：deposit_001, deposit_002, loan_001, loan_002"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_loan_repayment",
            "description": "计算车贷产品的还款明细，包括本金、利息和总金额，支持多种车贷方案",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "贷款产品ID，需为车贷产品（如：loan_001）"
                    },
                    "loan_amount": {
                        "type": "integer",
                        "description": "贷款金额（元），需在产品规定的金额范围内"
                    },
                    "years": {
                        "type": "number",
                        "description": "贷款年限，需为产品支持的期限选项"
                    },
                    "repayment_method": {
                        "type": "string",
                        "description": "还款方式，可选值：'等额本息'（默认）、'等额本金'",
                        "default": "等额本息"
                    }
                },
                "required": ["product_id", "loan_amount", "years"]
            }
        }
    }
]

# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_bank_products": get_bank_products,
    "get_product_detail": get_product_detail,
    "calculate_loan_repayment": calculate_loan_repayment,
}


def run_agent(user_query: str, api_key: str = "sk-a8da32a25ad54334af18a7c993804ccf", model: str = "qwen-plus"):
    """
    运行Agent，处理用户查询

    Args:
        user_query: 用户输入的问题
        api_key: API密钥（如果不提供则从环境变量读取）
        model: 使用的模型名称
    """
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key or os.getenv("sk-a8da32a25ad54334af18a7c993804ccf"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的汽车贷款助手。你可以：
            1. 介绍各种银行产品及其详细信息
            2. 根据客户需求计算不同产品的利息本金及总金额
            3. 根据客户需求计算贷款还款金额（等额本息/等额本金）
            4. 比较不同银行产品的收益或还款差异
            
            请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n" + "=" * 60)
    print("【用户问题】")
    print(user_query)
    print("=" * 60)

    # Agent循环：最多进行5轮工具调用
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        # 调用大模型
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让模型自主决定是否调用工具
        )

        response_message = response.choices[0].message
        print(f"LLM返回的消息：{response_message}")

        # 将模型响应加入对话历史
        messages.append(response_message)

        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("=" * 60)
            return response_message.content

        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False, indent=2)}")

            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                print(f"工具返回: {function_response[:200]}..." if len(
                    function_response) > 200 else f"工具返回: {function_response}")

                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                print(f"错误：未找到工具 {function_name}")

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"


# ==================== 示例场景 ====================

def demo_scenarios():
    """
    演示几个典型场景
    """
    print("\n" + "#" * 60)
    print("# 银行存贷产品Agent演示 - Function Call能力展示")
    print("#" * 60)

    scenarios = [
        "你们有哪些贷款产品？",
        "我想了解一下产品的详细信息",
        "我零首付怎么做最省钱",
        "帮我计算一下零利息的车贷方案"
        "给我一个全款三十五万零利息的车贷方案"
    ]

    print("\n以下是几个示例场景，您可以选择其中一个运行：\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")

    print("\n" + "-" * 60)
    print("要运行示例，请取消注释main函数中的相应代码")
    print("并确保设置了环境变量：DASHSCOPE_API_KEY")
    print("-" * 60)


if __name__ == "__main__":
    # 展示示例场景
    run_agent("给我一个全款三十五万零利息的车贷方案", model="qwen-plus")

    # 自定义查询
    # run_agent("你的问题", model="qwen-plus")
