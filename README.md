# KeywordCall

适配xxxbot-pad版本的KeywordCall，支持配置关键字进行不同API的调用，兼容OpenAI格式的API接口
安装后请注意修改config.toml的配置内容

# config.toml配置说明

```bash
[KeywordCall.keywords."#query"]  #query为关键字
open_ai_api_url = "https://api.openai.com/v1"   API的基础URL
api-key =  "sk-cxxcxxxxxxx"   鉴权key
open_ai_model =  "gpt-4o"    调用模型
prompt = "你是一个优秀的图片生成专家，你负责根据客户的要求生成图片"   提示词
image_regex = "https://[^\\s\\)]+?\\.png"     提取图片的正则表达式
```

# 使用说明

输入#query+内容

<div align="center">
<img width="700" src="./docs/1746972108648.jpg">
</div>
