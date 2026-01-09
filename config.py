# 配置：学科分类 + 规则阈值
import os

# （可选）本地开发用 .env：pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
# ===== 一、学科分类：与你截图里的表格保持一致 =====

CATEGORIES = [
    "人文科学类",
    "社会科学类",
    "理工类",
    "语言教育类",
    "艺术类",
    "其他类",
]

QUOTA_FACTOR_CAI = 1.15  # 只对“采”做配额限制  1.1~1.3 就改这里

# 逻辑角色 -> 具体模型名称（和 get_llm 参数对应）
LLM_ROUTES = {
    "router": "deepSeek",              # 分类路由智能体用哪个模型
    "subject_humanities": "deepSeek",  # 人文学科评估
    "subject_social": "deepSeek",
    "subject_stem": "tongyi",
    "subject_language": "tongyi",
    "subject_arts": "tongyi",
    "subject_other": "deepSeek",
    "policy": "hunyuan",              # 最终 policy 审查用混元
    "default": "local",
    # 你也可以加更多角色，比如 "debug" 等
}

LLM_key = {
    # ===== OpenAI 官方（GPT-4o / GPT-4o mini / GPT-3.5 等）=====
    "openai": os.getenv("OPENAI_API_KEY"),
    # 可选：如果走代理/网关（兼容 OpenAI），填这个；不用就留空
    "openai_base_url": os.getenv("OPENAI_BASE_URL"),

    # ===== DeepSeek =====
    # 你之前用的是 LLM_key.get("deepSeek")，建议保留这个键名以兼容旧代码
    "deepSeek": os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY"),

    # ===== 通义千问（DashScope / Tongyi）=====
    "tongyi": os.getenv("DASHSCOPE_API_KEY") or os.getenv("TONGYI_API_KEY"),

    # ===== 腾讯混元 =====
    "hunyuan": os.getenv("HUNYUAN_API_KEY"),
    "hunyuan_base_url": os.getenv("HUNYUAN_BASE_URL") or "https://api.hunyuan.cloud.tencent.com/v1",

    # ===== 本地 Ollama（跑 Llama/Qwen/DeepSeek-R1 本地模型）=====
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434",

    # ===== OpenAI-compatible 平台（可选项：你用哪个就配哪个）=====
    # OpenRouter
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "openrouter_base_url": os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",

    # Groq（如果你用）
    "groq": os.getenv("GROQ_API_KEY"),
    "groq_base_url": os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1",

    # Together（如果你用）
    "together": os.getenv("TOGETHER_API_KEY"),
    "together_base_url": os.getenv("TOGETHER_BASE_URL") or "https://api.together.xyz/v1",

    # Fireworks（如果你用）
    "fireworks": os.getenv("FIREWORKS_API_KEY"),
    "fireworks_base_url": os.getenv("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1",

    # 自建 vLLM / One-API / 任何 OpenAI-compatible 网关（推荐统一用这一组）
    "OPENAI_COMPAT_API_KEY": os.getenv("OPENAI_COMPAT_API_KEY"),
    "OPENAI_COMPAT_BASE_URL": os.getenv("OPENAI_COMPAT_BASE_URL"),
}




# ========= 配额控制 =========
# 可选： "rate"（按比例） / "fixed"（固定数量） / "off"（关闭）
QUOTA_MODE = "off"

# 配额基数：建议用“未命中馆藏”的候选数（更贴近真实可采购集合）
QUOTA_BASE = "remaining"   # "remaining" or "all"

# ——方式1：按比例——
MAX_CAI_RATE = 0.06        # “采”最多占候选的 6%（建议 5%~8%）
MAX_REC_RATE = 0.15        # “采+建议采”最多占候选的 15%（建议 10%~20%）
ENABLE_REC_LIMIT = True    # 是否对“采+建议采”也做总量限制（超额降级为复核）

# ——方式2：固定数量——
MAX_CAI_ABS = 300          # “采”最多 300 条
MAX_REC_ABS = 800          # “采+建议采”最多 800 条（其余降级为复核）

# 画像文件路径 & 指标
PROFILE_JSON_PATH = "profiles/outputs/馆藏与借阅画像_profile.json"
PROFILE_METRIC = "借书总次数"

# ========= 评估用（可选，离线对齐“已订”才用） =========
# 你如果还想保留旧的“按已订量*系数”用于离线回放评估，可留着
QUOTA_FACTOR_CAI = 1.2








# 各大类包含的学院（来自表格）
CATEGORY_COLLEGES = {
    "人文科学类": ["文学院", "历史文化学院", "马克思主义学院", "齐鲁文化研究院"],
    "社会科学类": ["经济学院", "法学院", "公共管理学院", "商学院", "心理学院", "新闻与传媒学院"],
    "理工类": [
        "信息科学与工程学院", "物理与电子科学学院", "数学与统计学院",
        "化学化工与材料科学学院", "地理与环境学院", "生命科学学院", "环境生态研究院",
    ],
    "语言教育类": ["外国语学院", "国际教育学院", "教育学部", "体育学院"],
    "艺术类": ["美术学院", "音乐学院", "赫尔岑国际艺术学院"],
}

DEFAULT_CATEGORY = "其他类"

# ===== 二、推荐规则参数（给 RulesAgent 用） =====

MAX_PRICE = 300.0          # 单册价格上限，可按学校规定调
MIN_PUB_YEAR = 2018        # 最早出版年，太老的默认不采
FORBIDDEN_KEYWORDS = [     # 敏感/不采主题关键词（示例）
    "成人", "博彩", "低俗",
]

# 对“推荐等级”统一一下枚举，便于后面处理
LEVEL_ACCEPT = "必选"
LEVEL_OPTIONAL = "可选"
LEVEL_REJECT = "不推荐"
LEVEL_REVIEW = "需人工复核"



# ===== 三、图书馆选书原则：规则配置 =====

# 1. 装帧类型中“一般不要”的
EXCLUDE_BINDING_TYPES = ["散页", "折页", "线装", "活页"]

# 2. 题名/简介中命中就基本不要的关键词
EXCLUDE_TITLE_KEYWORDS = [
    "资格考试", "职称考试", "考试大纲", "冲刺", "真题", "历年真题",
    "模拟试题", "考点精讲", "押题", "辅导", "题库", "习题集",
    "连环画", "漫画", "宣传画", "绘本",
    "微课", "课件",
]

# 3. 出版社“一般不要”
EXCLUDE_PUBLISHERS = [
    "国家开放大学出版社",          # 6
    "××电子音像出版社",            # 9 —— 这里可以列出具体几家
]

# 4. 主题/学科上“尽量不要或少买”的关键词（在 CLC 或题名中判断）
EXCLUDE_SUBJECT_KEYWORDS = [
    "高职", "高专",                 # 1
    "施工组织", "施工技术", "土木工程", "水利工程",  # 8
]

# 经济类整体偏谨慎（12）
ECONOMICS_CLC_PREFIX = ["F"]  # CLC 以 F 开头大多属经济类

# 法律整体慎选，但“法律史/理论”放宽（13）
LAW_CLC_PREFIX = ["D9"]       # 示例，可根据你馆的实际再细化
LAW_GOOD_KEYWORDS = ["法律史", "法制史", "法哲学", "法理学", "法律理论"]

# 马克思主义图书选优质出版社（14）
MARX_KEYWORDS = ["马克思", "恩格斯", "资本论"]
MARX_GOOD_PUBLISHERS = ["人民出版社", "高等教育出版社"]

# 民俗类：大城市民俗可以优先（15）
FOLK_KEYWORDS = ["民俗", "民间文化", "风俗"]
BIG_CITY_KEYWORDS = ["上海", "北京", "广州", "深圳", "天津", "重庆", "香港", "澳门"]

# 教育/体育类：选优质（16）
EDU_SPORT_KEYWORDS = ["教育", "教学", "体育", "运动训练"]
EDU_SPORT_GOOD_PUBLISHERS = ["人民教育出版社", "高等教育出版社", "北京师范大学出版社"]

# 建筑类：只要可持续、绿色理论（17）
ARCH_KEYWORDS = ["建筑", "城市设计"]
GREEN_KEYWORDS = ["可持续", "绿色建筑", "生态", "低碳", "节能"]

# 山东、齐鲁、乡村/县、地方志等优先&可多册（10）
LOCAL_KEYWORDS = ["山东", "齐鲁", "乡村", "村志", "县志", "地方志"]




# ===== 硬规则：一目了然就可以直接筛掉的情况 =====

# 1. 高职、高专等职业教育类（题名/简介/读者对象）
VOCATIONAL_KEYWORDS = [
    "高职", "高专", "职业学院", "职业教育",
    "中职", "中等职业", "技工学校","职业院校", "技工",
]

# 2. 连环画、漫画、绘本等
COMIC_KEYWORDS = [
    "连环画", "漫画", "绘本", "宣传画",
]

# 3. 各类资格考试/辅导类
EXAM_BOOK_KEYWORDS = [
    "资格考试", "职称考试", "公务员考试",
    "考试大纲", "冲刺", "押题", "真题",
    "历年真题", "模拟试题", "题库", "习题集",
    "辅导", "精讲精练",
]

# 4. 微课类
MICRO_COURSE_KEYWORDS = [
    "微课", "微课程", "慕课微课", "微视频课程",
]

# 5. 施工、土木、水利工程等（题名/简介/分类号）
CONSTRUCTION_KEYWORDS = [
    "施工", "土木工程", "结构工程", "市政工程",
    "水利工程", "水工结构", "工程施工", "施工技术",
]

# 6. 旅游厅、政府等出版物（出版社/出版单位）
GOV_TOURISM_PUBLISHER_KEYWORDS = [
    "人民政府", "市政府", "省政府",
    "文化和旅游厅", "旅游局", "旅游委",
    "××市人民政府", "××县人民政府",  # 可以写具体的
]

# 7. 国家开放大学出版社
OPEN_UNIV_PUBLISHER_KEYWORDS = [
    "国家开放大学出版社",
]

# 8. 电子音像出版社（可以再补充具体几家）
AUDIOVIDEO_PUBLISHER_KEYWORDS = [
    "电子音像出版社",
    "音像出版社",
]

# 9. 开本、页数硬规则的阈值
MIN_BOOK_HEIGHT_CM = 19.0   # 小于这个高度一般不采
MIN_BOOK_PAGES = 80         # 少于这个页数一般不采



#每个学科有自己“优先出版社”“认可出版社”“不太想要的出版社”；
SUBJECT_PROFILES = {
    "人文科学类": {
        "preferred_publishers": [
            "中华书局", "商务印书馆", "人民文学出版社",
            "上海古籍出版社", "中华书局上海编辑所",
        ],
        "reputable_publishers": [
            "北京大学出版社", "复旦大学出版社", "南京大学出版社",
            "社会科学文献出版社",
        ],
        "avoid_publishers": [
            # 人文方向不太需要的，比如太偏应试/工具书的社，可按你馆实际补充
        ],
        "research_keywords": [
            "史", "考古", "文献", "经典研究", "理论研究",
            "批评", "思想史", "哲学史", "比较研究",
        ],
        "low_value_keywords": [
            "心灵鸡汤", "随笔集", "故事会", "文学欣赏入门", "幽默笑话",
        ],
        "local_value_keywords": [
            "齐鲁文化", "山东地方文献", "地方志", "乡村文化", "县志", "市志",
        ],
    },

    "社会科学类": {
        "preferred_publishers": [
            "中国人民大学出版社", "社会科学文献出版社",
            "经济科学出版社", "江苏人民出版社",
        ],
        "reputable_publishers": [
            "高等教育出版社", "中国社会科学出版社",
            "北京大学出版社",
        ],
        "avoid_publishers": [],
        "research_keywords": [
            "实证研究", "理论研究", "比较研究", "案例研究",
            "问卷调查", "数据分析", "计量", "模型",
        ],
        "low_value_keywords": [
            "成功学", "励志故事", "财富自由", "轻松理财",
        ],
        "local_value_keywords": [
            "山东省经济", "山东社会发展", "区域发展", "黄河流域",
        ],
    },

    "理工类": {
        "preferred_publishers": [
            "科学出版社", "高等教育出版社",
            "机械工业出版社", "电子工业出版社",
        ],
        "reputable_publishers": [
            "清华大学出版社", "北京大学出版社",
        ],
        "avoid_publishers": [],
        "research_keywords": [
            "理论研究", "算法", "模型", "实验研究", "仿真",
            "前沿技术", "高级教程", "研究生教材",
        ],
        "low_value_keywords": [
            "入门速成", "完全教程", "零基础学会", "初学者指南",
        ],
        "local_value_keywords": [],
    },

    "语言教育类": {
        "preferred_publishers": [
            "外语教学与研究出版社", "上海外语教育出版社",
            "人民教育出版社",
        ],
        "reputable_publishers": [
            "北京师范大学出版社", "高等教育出版社",
        ],
        "avoid_publishers": [],
        "research_keywords": [
            "教育研究", "课程研究", "教学论", "教师发展",
            "语言学理论", "二语习得", "读写教育",
        ],
        "low_value_keywords": [
            "口语大全", "旅游英语口语", "零基础口语", "速成英语",
        ],
        "local_value_keywords": [
            "山东教育", "基础教育改革", "教师教育", "师范生教育",
        ],
    },

    "艺术类": {
        "preferred_publishers": [
            "人民音乐出版社", "中央音乐学院出版社",
            "湖南美术出版社", "江苏美术出版社",
        ],
        "reputable_publishers": [
            "中国青年出版社", "中国戏剧出版社",
        ],
        "avoid_publishers": [],
        "research_keywords": [
            "艺术史", "美学", "作品研究", "表演研究",
            "创作理论", "艺术教育",
        ],
        "low_value_keywords": [
            "简笔画入门", "填色本", "儿童涂色",
        ],
        "local_value_keywords": [
            "山东民间艺术", "齐鲁民间音乐", "地方戏曲",
        ],
    },

    "其他类": {
        "preferred_publishers": [],
        "reputable_publishers": [],
        "avoid_publishers": [],
        "research_keywords": ["研究", "理论", "方法", "前沿"],
        "low_value_keywords": ["漫画", "故事", "轻松读本"],
        "local_value_keywords": ["山东", "齐鲁"],
    },
}


