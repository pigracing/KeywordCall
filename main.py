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
from urllib.parse import urlparse, unquote
import subprocess
import shlex

@dataclass
class ModelConfig:
    open_ai_api_url: str
    api_key: str
    open_ai_model: str
    call_type: str  #openai,cmd
    prompt: str
    image_regex: str
    response_type: str #text,image,video,gemini
    is_translate: bool

class KeywordCall(PluginBase):
    description = "关键字调用"
    author = "pigracing"
    version = "1.0.1"

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
                call_type=_config.get("call_type","openai"),
                prompt=_config["prompt"],
                response_type = _config.get("response_type","text"),
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
        _config = self.keywords[matched_name]
        out_messages = []
        try:
            if _config.call_type == "openai":
                out_messages = await self.call_openai_api(_config, [{"role":"system","content":_config.prompt},{"role": "user", "content": content}])
            else:
                out_messages = await self.execute_curl_command(_config, [{"role":"system","content":_config.prompt},{"role": "user", "content": content}])
            logger.debug("返回内容: " + out_messages[0][:200])
            response_type = _config.response_type
            for out_message in out_messages:
                if response_type == 'image' or response_type == 'gemini':
                    
                    if self.is_image_url(out_message):
                        # 如果返回的是图片链接，直接发送
                        base64_str = await self.image_url_to_base64(out_message)
                        logger.debug("图片链接转换为base64: " + base64_str[:100])
                        await bot.send_image_message(message["FromWxid"], base64_str)
                    elif out_message.startswith("data:image"):
                        str_arr = out_message.split(",")
                        logger.debug("图片链接转换为base64: " + str_arr[1][:100])
                        await bot.send_image_message(message["FromWxid"], str_arr[1])
                    else:
                        await bot.send_text_message(message["FromWxid"], out_message)
                elif response_type == 'video':
                    base64_str = await self.image_url_to_base64(out_message)
                    # 发送视频消息
                    await bot.send_video_message(message["FromWxid"], base64_str)
                else:  
                    # 如果返回的是文本，直接发送
                    await bot.send_text_message(message["FromWxid"], out_message)
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await bot.send_text_message(message["FromWxid"], "处理消息失败，请稍后再试。")
        return False  # 阻止后续插件处理
    
    async def execute_curl_command(self,config: ModelConfig, messages: list[Dict[str, Any]]) -> str:
        print("正在通过调用系统 curl 命令发送请求...")

        # 1. 将 Python 字典转换为 JSON 字符串
        #    这是 curl -d 参数需要的内容
        payload = {
            "model": config.open_ai_model,
            "stream": False,
            "messages": messages,
            "temperature": 0.7
        }
        data_string = json.dumps(payload)

        # 2. 构建 curl 命令
        #    我们使用列表来构建命令，这是更安全的方式，可以避免 shell 注入风险
        #    每个参数和它的值都作为列表中的独立元素
        command = [
            'curl',
            '--max-time', '620',  # 设置最大总时间
            '-X', 'POST',          # 请求方法
            
            # 设置 Headers
            '-H', f'Authorization: Bearer {config.api_key}',
            '-H', 'Content-Type: application/json',
            '-H', 'Connection: keep-alive',
            
            # 设置请求体
            '-d', data_string,
            
            # 目标 URL
            f"{config.open_ai_api_url}/chat/completions"
        ]
        
        # 打印将要执行的命令，方便调试
        # shlex.join 在 Python 3.8+ 中可用，可以安全地将列表组合成一个可读的命令行字符串
        try:
            print(f"执行命令: {shlex.join(command)}")
        except AttributeError:
            # 兼容旧版本 Python
            print(f"执行命令: {' '.join(shlex.quote(s) for s in command)}")

        try:
            # 3. 执行命令
            #    - capture_output=True: 捕获标准输出和标准错误
            #    - text=True: 以文本模式（自动解码）返回输出，如果为False则返回bytes
            #    - check=True: 如果命令返回非零退出码（即出错），则抛出 CalledProcessError 异常
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True,  # 使用 text=True，stdout 和 stderr 会是字符串
                check=True,
                encoding='utf-8' # 明确指定编码
            )

            # 4. 处理成功的结果
            print("curl 命令执行成功！")
            
            # curl 的输出在 result.stdout 中
            response_body = result.stdout
            
            # 尝试将返回的字符串解析为 JSON
            try:
                data = json.loads(response_body)
                # 解析内容
                if config.response_type == 'gemini':
                    text = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
                else:
                    text = data["choices"][0]["message"]["content"]

                print("resp:"+text[0:100])

                # 使用正则提取（如配置了 image_regex）
                if config.image_regex:
                    matches = re.findall(config.image_regex, text)
                    return matches
                    #if matches:
                    #    return "\n".join(matches)
                return [text]
            except json.JSONDecodeError:
                print("响应不是有效的 JSON 格式，返回原始文本。")
                return response_body

        except FileNotFoundError:
            print("错误：'curl' 命令未找到。请确保 curl 已安装并在系统的 PATH 中。")
            return None
        except subprocess.CalledProcessError as e:
            # 如果 curl 执行失败 (e.g., 网络错误, HTTP 4xx/5xx 错误)
            print(f"curl 命令执行失败，返回码: {e.returncode}")
            print("--- 标准输出 (stdout) ---")
            print(e.stdout)
            print("--- 标准错误 (stderr) ---")
            print(e.stderr)
            return None
        except Exception as e:
            print(f"执行过程中发生未知错误: {e}")
            return None
    

    async def call_openai_api(self,config: ModelConfig, messages: list[Dict[str, Any]]) -> str:
        url = f"{config.open_ai_api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive"
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
                    print(text[0:100])
                    # 使用正则提取（如配置了 image_regex）
                    if config.image_regex:
                        matches = re.findall(config.image_regex, text)
                        #if matches:
                        #    return "\n".join(matches)
                        return matches
                    return [text]
        except Exception as e:
            logger.error(f"调用 OpenAI API 失败: {e}")
            return "调用 OpenAI API 失败，请稍后再试。"

    def is_image_url(self,url: str) -> bool:
        if not url.lower().startswith("http"):
            return False
        pattern = r'^https?://.*\.(jpg|jpeg|png|gif|bmp|webp|svg|tiff)(\?.*)?$'
        return bool(re.match(pattern, url.lower()))
    
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
