import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger("Scheduler")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"

DEFAULT_TIMED_MESSAGES = [
    {
        "id": "t1",
        "type": "interval",
        "interval_min": 10,
        "voice": "ron",
        "message": "!ron Welcome everyone to the stream! Enjoy the broadcast.",
        "enabled": True,
        "next_fire": time.time() + 600,
        "last_run": "Never"
    }
]

class StreamScheduler:
    def __init__(self, fire_callback: Optional[Callable] = None):
        self.fire_callback = fire_callback
        self.timed_messages: List[Dict[str, Any]] = []
        self.event_triggers: List[Dict[str, Any]] = []
        self.activity_log: deque = deque(maxlen=150)
        self.running: bool = False
        self._task: Optional[asyncio.Task] = None
        self.load()

    def load(self):
        if CONFIG_FILE.exists():
            try:
                cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                self.timed_messages = cfg.get("timed_messages", DEFAULT_TIMED_MESSAGES)
                self.event_triggers = cfg.get("event_triggers", [])
            except Exception as e:
                logger.warning(f"Error loading scheduler config: {e}")
                self.timed_messages = DEFAULT_TIMED_MESSAGES
        else:
            self.timed_messages = DEFAULT_TIMED_MESSAGES

    def save(self):
        if CONFIG_FILE.exists():
            try:
                cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                cfg["timed_messages"] = self.timed_messages
                cfg["event_triggers"] = self.event_triggers
                CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning(f"Error saving scheduler config: {e}")

    def log_activity(self, text: str, source: str = "System"):
        entry = {
            "id": f"act_{int(time.time()*1000)}",
            "time": time.strftime("%H:%M:%S"),
            "source": source,
            "text": text
        }
        self.activity_log.appendleft(entry)

    def start(self):
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._loop())
            logger.info("✓ StreamScheduler background worker started.")

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("StreamScheduler stopped.")

    async def _loop(self):
        while self.running:
            now = time.time()
            for msg in self.timed_messages:
                if not msg.get("enabled", True):
                    continue

                m_type = msg.get("type", "interval")
                if m_type == "interval":
                    next_fire = msg.get("next_fire", 0)
                    if now >= next_fire:
                        interval_sec = int(msg.get("interval_min", 10)) * 60
                        msg["next_fire"] = now + interval_sec
                        msg["last_run"] = time.strftime("%H:%M:%S")
                        self.save()

                        if self.fire_callback:
                            try:
                                voice = msg.get("voice", "ron").replace("!", "").strip()
                                raw_msg = msg.get("message", "").strip()
                                text_to_fire = raw_msg if raw_msg.startswith("!") else f"!{voice} {raw_msg}"
                                await self.fire_callback(text_to_fire, "Scheduled")
                            except Exception as err:
                                logger.error(f"Error firing scheduled interval announcement: {err}")

                elif m_type == "at_time":
                    target_time = msg.get("time", "12:00")
                    current_hhmm = time.strftime("%H:%M")
                    last_run_date = msg.get("last_run_date", "")
                    today_date = time.strftime("%Y-%m-%d")

                    if current_hhmm == target_time and last_run_date != today_date:
                        msg["last_run_date"] = today_date
                        msg["last_run"] = time.strftime("%H:%M:%S")
                        self.save()

                        if self.fire_callback:
                            try:
                                voice = msg.get("voice", "ron").replace("!", "").strip()
                                raw_msg = msg.get("message", "").strip()
                                text_to_fire = raw_msg if raw_msg.startswith("!") else f"!{voice} {raw_msg}"
                                await self.fire_callback(text_to_fire, "Scheduled")
                            except Exception as err:
                                logger.error(f"Error firing exact time announcement: {err}")

            await asyncio.sleep(5.0)

scheduler = StreamScheduler()
