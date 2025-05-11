import tomllib
import aiohttp
import asyncio
import base64
import mimetypes
import re
from typing import Any, Dict
from loguru import logger
from dataclasses import dataclass, field
import json

from WechatAPI import WechatAPIClient
from utils.decorators import *
from utils.plugin_base import PluginBase

@dataclass
class ModelConfig:
    open_ai_api_url: str
    api_key: str
    open_ai_model: str
    prompt: str
    image_regex: str
    is_translate: bool

class KeywordCall(PluginBase):
    description = "关键字调用"
    author = "pigracing"
    version = "1.0.0"

    def __init__(self):
        super().__init__()

        with open("plugins/KeywordCall/config.toml", "rb") as f:
            plugin_config = tomllib.load(f)

        config = plugin_config["KeywordCall"]
        self.enable = config["enable"]
        # 加载所有模型配置
        self.keywords = config.get("keywords", {})
        logger.debug(f"加载模型配置1: {self.keywords}")
        for _keyword, _config in self.keywords.items():
            logger.debug(f"加载模型配置2: {_config}")
            self.keywords[_keyword] = ModelConfig(
                open_ai_api_url=_config["open_ai_api_url"],
                api_key=_config["api-key"],
                open_ai_model=_config["open_ai_model"],
                prompt=_config["prompt"],
                image_regex=_config.get("image_regex",""),
                is_translate=_config.get("is_translate",False)
            )
        logger.debug(f"加载模型配置3: {self.keywords}")
    
    def match_keyword_name(self, text: str) -> str | None:
        for _keyword in self.keywords:
            if text.startswith(_keyword):
                return _keyword
        return None

    @on_text_message
    async def handle_text(self, bot: WechatAPIClient, message: dict):
        if not self.enable:
            return

        matched_name = self.match_keyword_name(message["Content"])
        if matched_name:
            logger.debug(f"匹配到关键字: {matched_name}")
        else:
            logger.debug("没有匹配到关键字")
            return
        
        content = message["Content"]
        content = content[len(matched_name):].strip()
        logger.debug("处理内容: " + content)
        try:
            out_message = await self.call_openai_api(self.keywords[matched_name], [{"role": "user", "content": content}])
            logger.debug("返回内容: " + out_message)
            if self.is_image_url(out_message):
                # 如果返回的是图片链接，直接发送
                base64_str = await self.image_url_to_base64(out_message)
                logger.debug("图片链接转换为base64: " + base64_str[:100])
                await bot.send_image_message(message["FromWxid"], base64_str)
            else:  
                # 如果返回的是文本，直接发送
                await bot.send_text_message(message["FromWxid"], out_message)
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await bot.send_text_message(message["FromWxid"], "处理消息失败，请稍后再试。")
        return False  # 阻止后续插件处理
    

    async def call_openai_api(self,config: ModelConfig, messages: list[Dict[str, Any]]) -> str:
        url = f"{config.open_ai_api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config.open_ai_model,
            "stream": False,
            "messages": messages,
            "temperature": 0.7
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    response_text = await response.text()

                    if response.status != 200:
                        raise RuntimeError(f"OpenAI API 请求失败: {response.status} - {response_text}")

                    try:
                        data = json.loads(response_text)
                    except json.JSONDecodeError:
                        raise RuntimeError(f"响应无法解析为 JSON（Content-Type: {response.headers.get('Content-Type')}）：\n{response_text}")

                    # 解析内容
                    text = data["choices"][0]["message"]["content"]

                    # 使用正则提取（如配置了 image_regex）
                    if config.image_regex:
                        matches = re.findall(config.image_regex, text)
                        if matches:
                            return "\n".join(matches)

                    return text
        except Exception as e:
            logger.error(f"调用 OpenAI API 失败: {e}")
            return "调用 OpenAI API 失败，请稍后再试。"

    def is_image_url(self,url: str) -> bool:
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff')
        return url.lower().startswith("http") and url.lower().endswith(image_extensions)
    
    async def image_url_to_base64(self,url: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise Exception(f"图片下载失败: HTTP {resp.status}")
                    content = await resp.read()
                    base64_str = base64.b64encode(content).decode("utf-8")
                    return base64_str

        except Exception as e:
            return f"[图片处理错误] {str(e)}"
